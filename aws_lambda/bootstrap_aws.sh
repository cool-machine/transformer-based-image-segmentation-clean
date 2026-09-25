#!/usr/bin/env bash
set -euo pipefail

region="us-east-1"
stack_name="ocp8"
template_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ocp8-stack.yaml"

if ! aws cloudformation describe-stacks \
  --stack-name "$stack_name" \
  --region "$region" >/dev/null 2>&1; then
  printf '%s\n' \
    "The OCP8 stack does not exist. This helper will not recreate it from an empty ECR repository." \
    "Restore an OCP8 image first, then deploy aws_lambda/ocp8-stack.yaml deliberately."
  exit 1
fi

aws cloudformation deploy \
  --stack-name "$stack_name" \
  --region "$region" \
  --template-file "$template_file" \
  --capabilities CAPABILITY_NAMED_IAM \
  --no-fail-on-empty-changeset
