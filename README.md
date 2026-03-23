# Image Segmentation System

Production-focused semantic segmentation app with:
- GitHub Pages frontend (`index.html`)
- Azure Functions backend (`backend/function_app.py`)
- Azure Blob Storage image/mask source

## Live Architecture

- **Frontend**: static page on GitHub Pages
- **Backend**: Azure Functions HTTP API
- **Storage**: Azure Blob container `images1`

## Live URLs

- **Frontend**: `https://cool-machine.github.io/transformer-based-image-segmentation-clean/`
- **Backend API**: `https://ocp8-centralus-v2.azurewebsites.net/api`
- **Backend health**: `https://ocp8-centralus-v2.azurewebsites.net/api/health`

## API Contract Used by Frontend

The frontend calls only these endpoints:
- `GET /api/health`
- `GET /api/images`
- `GET /api/image-thumbnail?image_name=...`
- `GET /api/colorized-masks?image_name=...`

## Important Runtime Behavior

- Model inference is **required** for `colorized-masks`.
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

### Backend (Azure Functions)

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
- **Azure Functions**: `.github/workflows/deploy-functions.yml`
- **Published frontend**: `https://cool-machine.github.io/transformer-based-image-segmentation-clean/`
- **Published backend**: `https://ocp8-centralus-v2.azurewebsites.net/api`

## Notes

- `requirements.txt` at repo root delegates to `backend/requirements.txt`.
- The backend workflow installs dependencies from `backend/requirements.txt`.
