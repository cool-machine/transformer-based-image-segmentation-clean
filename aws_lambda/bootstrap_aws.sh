#!/usr/bin/env bash
set -euo pipefail

region="us-east-1"
account_id="$(aws sts get-caller-identity --query Account --output text)"
repository="ocp8-segmentation"
function_name="ocp8-segmentation"
bucket="clarifiance-ocp8-artifacts-174208891400"
lambda_role="ocp8-lambda-execution"
deploy_role="github-ocp8-deploy"
oidc_url="https://token.actions.githubusercontent.com"
oidc_arn="arn:aws:iam::${account_id}:oidc-provider/token.actions.githubusercontent.com"

if ! aws iam get-open-id-connect-provider \
  --open-id-connect-provider-arn "$oidc_arn" >/dev/null 2>&1; then
  aws iam create-open-id-connect-provider \
    --url "$oidc_url" \
    --client-id-list sts.amazonaws.com >/dev/null
fi

lambda_trust="$(mktemp)"
cat >"$lambda_trust" <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
JSON

if ! aws iam get-role --role-name "$lambda_role" >/dev/null 2>&1; then
  aws iam create-role \
    --role-name "$lambda_role" \
    --assume-role-policy-document "file://$lambda_trust" >/dev/null
fi

lambda_policy="$(mktemp)"
cat >"$lambda_policy" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "logs:CreateLogGroup",
      "Resource": "arn:aws:logs:${region}:${account_id}:*"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:${region}:${account_id}:log-group:/aws/lambda/${function_name}:*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${bucket}",
      "Condition": {"StringLike": {"s3:prefix": ["models/*", "images1/*"]}}
    },
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::${bucket}/models/*",
        "arn:aws:s3:::${bucket}/images1/*"
      ]
    }
  ]
}
JSON
aws iam put-role-policy \
  --role-name "$lambda_role" \
  --policy-name ocp8-runtime-access \
  --policy-document "file://$lambda_policy"

github_trust="$(mktemp)"
cat >"$github_trust" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "${oidc_arn}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:cool-machine/transformer-based-image-segmentation-clean:ref:refs/heads/main"
      }
    }
  }]
}
JSON

if ! aws iam get-role --role-name "$deploy_role" >/dev/null 2>&1; then
  aws iam create-role \
    --role-name "$deploy_role" \
    --assume-role-policy-document "file://$github_trust" >/dev/null
else
  aws iam update-assume-role-policy \
    --role-name "$deploy_role" \
    --policy-document "file://$github_trust"
fi

deploy_policy="$(mktemp)"
cat >"$deploy_policy" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:GetDownloadUrlForLayer",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart"
      ],
      "Resource": "arn:aws:ecr:${region}:${account_id}:repository/${repository}"
    },
    {
      "Effect": "Allow",
      "Action": [
        "lambda:AddPermission",
        "lambda:CreateFunction",
        "lambda:CreateFunctionUrlConfig",
        "lambda:GetFunction",
        "lambda:GetFunctionConfiguration",
        "lambda:GetFunctionUrlConfig",
        "lambda:UpdateFunctionCode"
      ],
      "Resource": "arn:aws:lambda:${region}:${account_id}:function:${function_name}"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::${account_id}:role/${lambda_role}",
      "Condition": {"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}}
    }
  ]
}
JSON
aws iam put-role-policy \
  --role-name "$deploy_role" \
  --policy-name ocp8-github-deployment \
  --policy-document "file://$deploy_policy"

if ! aws ecr describe-repositories \
  --repository-names "$repository" \
  --region "$region" >/dev/null 2>&1; then
  aws ecr create-repository \
    --repository-name "$repository" \
    --region "$region" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 >/dev/null
fi

printf 'AWS_DEPLOY_ROLE_ARN=arn:aws:iam::%s:role/%s\n' "$account_id" "$deploy_role"
printf 'AWS_LAMBDA_ROLE_ARN=arn:aws:iam::%s:role/%s\n' "$account_id" "$lambda_role"
printf 'ECR_REPOSITORY=%s\n' "$repository"
