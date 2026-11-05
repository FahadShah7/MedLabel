# AI Usage

This file records how AI tools were used in the MedLabel project, as required by the course and promised in our Milestone 0 proposal. It covers Milestone 1 and will be updated for Milestone 2.

## Summary

We used **Claude (Anthropic)** as an assistant throughout Milestone 1. Claude drafted most of the code and documentation. The team ran every script, executed the tests, trained the models, checked the outputs and made the final decisions about what went into the repository. We did not write everything from scratch, and we do not claim to have.

The proposal said AI would be used for "boilerplate, debugging and documentation drafting", and that every AI-suggested change would be reviewed and understood by the group. In practice, AI was used a little more heavily for first drafts of code than that wording suggests, so we state it plainly here.

## Where AI was used

| Area | What the AI did | What we did |
|---|---|---|
| Project structure | Suggested the folder layout and the feature-branch workflow | Created the repo, branch `feat/api-predict`, and the commits |
| `src/prepare_data.py` | Drafted the script: dataset loading, validation checks, stratified 80/20 split with seed 42 | Ran it, checked the 800/200 split and the class mix (55% yes, 34% no, 11% maybe) |
| `src/train.py` | Drafted the TF-IDF + logistic regression pipeline, the five MLflow runs and the best-run selection | Ran the training in Google Colab, downloaded `mlflow.db`, inspected the runs in the MLflow UI |
| `src/app.py` | Drafted the FastAPI service: `/predict`, `/explain`, `/health`, `/metrics`, Pydantic validation, BM25 retrieval | Ran the service locally, tried the endpoints in `/docs` |
| `tests/` | Drafted the 12 pytest tests (4 data validation, 8 API) | Ran them with `make test` and confirmed all pass |
| Makefile and `requirements.txt` | Drafted the targets and suggested pinning `scikit-learn==1.6.1` | Ran `make setup`, `data`, `train`, `test`, `lint`, `run` |
| README | Drafted the text, results tables and limitations section | Checked the numbers against the MLflow runs and metrics JSON |
| Debugging | Helped diagnose a missing virtual environment, a zsh quirk that broke a `.gitignore` command, a truncated terminal paste and a hidden `.env.example` pattern | Applied the fixes and confirmed they worked |
| Linting | Suggested the fix for the one style issue ruff reported (`Optional[str]` to `str | None`) | Re-ran ruff until clean |
| Presentation | Helped build the Milestone 1 slide deck and a study guide | Reviewed the content against the repo |

## Where AI was not used

- **The data.** PubMedQA and its expert labels come from the dataset authors. AI did not create or alter any data.
- **Running experiments.** The training runs and the MLflow results are real outputs from our own runs. We did not take any numbers from the AI.
- **The decision to use a simple model.** This was made in the proposal so the project would focus on the pipeline.

## How we checked AI output

- Ran the code ourselves instead of trusting it. Tests passing and `make run` working were the bar, not how the code looked.
- Read the results rather than copying them. For example, we noticed that Naive Bayes predicts "yes" for nearly everything, and that the four logistic regression runs are within 0.015 macro-F1 of each other, which is probably noise on 200 test questions.
- Reviewed what was staged before committing. A temporary stand-in script (`scripts/make_dummy_model.py`) had been staged at one point and was removed. A wrong screenshot nearly got committed and was caught before it was.
- Checked the commit history with `git log --stat` and `git show 48762b2 --stat` so nothing stray remained.

## Known limits of the AI-assisted work

These are things we found or know about, and they are also listed in the README:

- Macro-F1 is modest (0.40), and "maybe" is weak (0.18).
- The best run was selected using the test set, so its score is slightly optimistic. A separate validation split would fix this.
- Training used each question's own abstract, but the API without a context uses retrieved passages that may come from a different abstract.
- In the demo, retrieval can find a dataset question's own abstract because the corpus is the same dataset.
- Confidence scores are not calibrated.
- `train.py` and retrieval quality are not covered by tests.

## Third-party ideas and libraries

- **PubMedQA** (dataset), loaded through the Hugging Face `datasets` library.
- **MedRAG**: the retrieval idea is reused from the earlier MedRAG project and credited in the README. No code was copied without attribution.
- **Libraries:** scikit-learn, MLflow, FastAPI, Pydantic, rank-bm25, pytest, ruff.

## Team members

Each member should add a short note on how they personally used AI. Edit your own line only.

| Member | ERP ID | AI use and what I verified myself |
|---|---|---|
| Ameer Hamza | 28308 | _add your note_ |
| Fahad Shah | 29255 | _add your note_ |
| Ahmed Hassan Khan | 29299 | _add your note_ |
