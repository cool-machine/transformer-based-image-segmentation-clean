# Image Segmentation System

Production-focused semantic segmentation recruiting demo with:
- GitHub Pages frontend (`index.html`)
- AWS Lambda container backend (`aws_lambda/handler.py`)
- Private Amazon S3 model/image/mask storage

## Live Architecture

- **Frontend**: static page on GitHub Pages
- **Backend**: AWS Lambda Function URL, scaling to zero
- **Storage**: private S3 bucket `clarifiance-ocp8-artifacts-174208891400`

## Live URLs

- **Frontend**: `https://cool-machine.github.io/transformer-based-image-segmentation-clean/`
- **Backend API**: set after the AWS deployment is verified
- **Legacy rollback API**: `https://ocp8-centralus-v2.azurewebsites.net/api`

## API Contract Used by Frontend

The frontend calls only these endpoints:
- `GET /health`
- `GET /images`
- `GET /image-thumbnail?image_name=...`
- `GET /colorized-masks?image_name=...`

## Important Runtime Behavior

- Model inference is **required** for `colorized-masks` and loads lazily after scale-to-zero.
- If model dependencies (`tensorflow`, `transformers`) are missing, the API returns a **500 error**.
- If model load/prediction fails, the API returns a **500 error**.
- No silent fallback success path is kept.

## Repository Structure (Current)

```text
.
├── index.html
├── backend/
│   ├── function_app.py
│   ├── host.json
│   ├── local.settings.json.template
│   └── requirements.txt
├── aws_lambda/
│   ├── Dockerfile
│   ├── handler.py
│   └── requirements.txt
├── notebooks/
├── .github/workflows/
│   ├── simple-deploy.yml
│   └── deploy-functions.yml
└── requirements.txt
```

## Local Development

### Frontend

Serve the root folder and open the page:

```bash
python -m http.server 8080
```

Open: `http://localhost:8080`

### Legacy backend (Azure Functions rollback)

```bash
cd backend
pip install -r requirements.txt
func start --port 7071
```

Set local settings from template:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "IMAGES_STORAGE_CONNECTION_STRING": "<your-storage-connection-string>"
  }
}
```

## Deployment

- **GitHub Pages**: `.github/workflows/simple-deploy.yml`
- **AWS Lambda**: `.github/workflows/deploy-aws-lambda.yml`
- **Legacy Azure rollback**: `.github/workflows/deploy-functions.yml`
- **Published frontend**: `https://cool-machine.github.io/transformer-based-image-segmentation-clean/`
- **AWS deployment authentication**: GitHub OIDC with short-lived credentials; no AWS access keys are stored in GitHub.

## Notes

- `requirements.txt` at repo root delegates to `backend/requirements.txt`.
- The backend workflow installs dependencies from `backend/requirements.txt`.
