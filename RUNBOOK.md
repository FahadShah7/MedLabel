# MedLabel Runbook

How to deploy, check, redeploy and roll back the MedLabel API.

## Where it runs

| Item | Value |
|---|---|
| Platform | Render web service `medlabel` (Docker runtime, free plan), defined in `render.yaml` |
| Source | `main` branch of this repository, built from `Dockerfile` |
| Images | Every push to `main` is also pushed by CI to `ghcr.io/fahadshah7/medlabel` as `sha-<commit>` and `latest`; release tags are pushed as `v1.0-milestone1`, `v2.0-milestone2`, ... |
| Secrets | `GEMINI_API_KEY`, set in the Render dashboard (Environment tab). Never committed. |
| Health | `GET /health` (Render health check and Docker `HEALTHCHECK`) |
| Monitoring | `GET /stats` (request count, error count, average latency, per-endpoint latency, predicted label counts) and the Render Logs tab (one JSON line per request) |

## First deployment

1. On [Render](https://render.com), choose **New > Blueprint** and select this repository. Render reads `render.yaml`.
2. When asked, paste the `GEMINI_API_KEY` value. Without it, `/predict` and `/explain` still work and `/decide` and `/answer` return 503.
3. Wait for the deploy to show **Live**, then run the checks below.

## Checking a deployment

```bash
URL=https://medlabel.onrender.com   # replace with the service URL
curl $URL/health
curl -X POST $URL/predict -H "Content-Type: application/json" \
  -d '{"question": "Does aspirin reduce the risk of heart attack?"}'
curl $URL/stats
```

`/health` must return `"status": "ok"` and `"model_loaded": true`. The free plan sleeps after 15 minutes without traffic, so the first request can take about a minute.

## Redeploying

- **Normal path:** merge a pull request into `main`. CI runs lint, tests and the Docker build; Render auto-deploys the new commit.
- **Manual:** Render dashboard > `medlabel` > **Manual Deploy > Deploy latest commit**.
- **Change config only** (for example, rotate `GEMINI_API_KEY`): edit it in the Environment tab and save. Render restarts the service with the same image.

## Rolling back

Use the first option that applies.

1. **Render rollback (fastest).** Dashboard > `medlabel` > **Events**, find the last good deploy and click **Rollback**. This redeploys that exact earlier build. Turn off auto-deploy until `main` is fixed, otherwise the next push redeploys the broken code.
2. **Run a previous Docker image tag.** Every commit on `main` has an image in GHCR:
   ```bash
   docker pull ghcr.io/fahadshah7/medlabel:v1.0-milestone1   # or :sha-<commit>
   docker run -d -p 8000:8000 --name medlabel ghcr.io/fahadshah7/medlabel:v1.0-milestone1
   ```
   On Render, the same tag can be deployed by creating a service of type **Existing Image** with that image URL.
3. **Revert the code.** `git revert <bad-commit>`, open a pull request and merge it. CI and auto-deploy ship the fix.
4. **Roll back only the model.** Both model versions are in the image. Set the environment variable `MODEL_PATH=models/medlabel_v1.joblib` in Render and save; the service restarts on the older model.

After any rollback, run the checks in "Checking a deployment" and confirm `/health` shows the expected `model_version`.

## Common problems

| Symptom | Likely cause | Fix |
|---|---|---|
| Deploy fails at startup with `Model artifact not found` | `MODEL_PATH` points to a file not in the image | Unset `MODEL_PATH` or point it at `models/medlabel_v2.joblib` |
| `/decide` or `/answer` returns 503 | `GEMINI_API_KEY` not set | Add it in the Environment tab |
| `/decide` or `/answer` returns 502 | Gemini API error or quota exhausted | Check Render logs; retry later or check quota in Google AI Studio |
| `/health` shows `corpus_size: 50` | PubMedQA download failed during the image build | Redeploy; the build logs show `PubMedQA download failed` |
| First request takes about a minute | Free plan was asleep | Expected; call `/health` once before a demo |
