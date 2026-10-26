# Deploying

Everything deploys from GitHub Actions to one Google Cloud project, through Workload Identity Federation. No key is
stored in the repository.

| Workflow | Trigger | Does |
|---|---|---|
| `tests` | push to `alignr/`, `api/`, `tests/`, `facts/` | Unit tests and the baseline set |
| `alignr` | an issue from the *New generation set* form, or a manual run | Drafts the set with Gemini, builds it with DITA-OT, opens draft pull requests, commits the run record |
| `gates` | a push to a draft pull request; a merge | Grammar, rules and open-questions statuses; the merge record |
| `deploy-api` | push to the engine, API or facts | Builds the image, smoke-tests it with the baseline, pushes it to Artifact Registry, deploys `alignr-api` to Cloud Run |
| `pages` | push to `main` | Builds the site: design document, the app, the DITA help (HTML5, PDF, `csh.json`) and these docs |
| `live-check` | after `deploy-api` or `pages`, or manually | Checks the live page, CORS and one live draft from a runner |

## One-time settings

- Repository variables `GCP_PROJECT`, `GCP_REGION`, `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`, `GCP_RUNTIME_SA`,
  `GCP_BUCKET`, `GCP_AR_REPO` (copied from MANIFEST), and the repository ID bound to the deploy service account.
- Pages source: GitHub Actions.
- Actions may create pull requests (Settings, Actions, General, Workflow permissions).
- After the first deploy, make the service public:
  `gcloud run services add-iam-policy-binding alignr-api --region=us-central1 --member=allUsers --role=roles/run.invoker`.

## Cost controls

The service runs 0 to 1 instances with request-based billing. Public drafts are capped at 20 a day
(`ALIGNR_DAILY_DRAFTS`). A TPU7x set is 5 to 15 Gemini calls; the measured cost of a set is in every run record and on
the page.
