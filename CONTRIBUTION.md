# Project Contribution Log

## Group Information

| ERP ID | Student Name | Role / Focus Area |
|---|---|---|
| **28308** | Ameer Hamza | Data Pipeline & Dataset Validation |
| **29255** | Fahad Shah | Model Training & MLflow Experiment Tracking |
| **29299** | Ahmed Hassan Khan | FastAPI Service, BM25 Retrieval & Testing |

---

## Member-to-Task Breakdown (Milestone 1)

| Member | Tasks & Deliverables | Associated Files / Artifacts |
|---|---|---|
| **Ameer Hamza**<br>`(28308)` | • Ingestion of PubMedQA (`pqa_labeled`) using Hugging Face `datasets`<br>• Implemented dataset validation rules (missing value checks, non-empty text, label verification)<br>• Stratified 80/20 train/test split generation with fixed random seed (42)<br>• Developed data validation test cases<br>• Dataset profiling and class balance analysis | • `src/prepare_data.py`<br>• `tests/test_prepare_data.py`<br>• `data/sample.csv` |
| **Fahad Shah**<br>`(29255)` | • Designed TF-IDF feature extraction pipelines (unigrams, bigrams, sublinear TF)<br>• Trained and evaluated baseline classifiers (Logistic Regression and Multinomial Naive Bayes)<br>• Configured MLflow experiment tracking across 5 model configurations<br>• Metric logging: Macro-F1 (primary), Accuracy, and per-class F1 (yes/no/maybe)<br>• Serialized the best performing model artifact and metrics summary | • `src/train.py`<br>• `models/medlabel_v1.joblib`<br>• `models/medlabel_v1_metrics.json`<br>• `docs/mlflow_runs.png` |
| **Ahmed Hassan Khan**<br>`(29299)` | • Built the FastAPI serving application with lifespan state management<br>• Implemented BM25Okapi passage retrieval for supporting context extraction<br>• Developed `/predict`, `/explain`, `/health`, and `/metrics` endpoints with Pydantic schemas<br>• Implemented HTTP request latency tracking middleware (`X-Response-Time-Ms`)<br>• Authored comprehensive API integration tests and `Makefile` task runners | • `src/app.py`<br>• `tests/test_api.py`<br>• `Makefile` |

---

## Contribution Summary

- **Work Distribution:** All group members contributed equally (~33.3% each) across their respective domains (Data Preparation, Model Training, and Service Deployment).
- **Attribution & Acknowledgments:**
  - The BM25 context retrieval approach is adapted from the earlier project [MedRAG](https://github.com/FahadShah7/MedRag).
  - Dataset credit: Jin et al. (2019), *PubMedQA: A Dataset for Biomedical Research Question Answering* (EMNLP-IJCNLP 2019).
