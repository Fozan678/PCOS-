# PCOS Prediction from Survey Data: Refined Analysis Pipeline

Exploratory machine learning analysis of a short online questionnaire on
polycystic ovary syndrome (PCOS). The pipeline estimates how much
information self-reported lifestyle, anthropometric, and symptom data carry
about a reported PCOS diagnosis, with the uncertainty that a small sample
requires.

> **Scope.** This is an exploratory, descriptive analysis. It is **not** a
> validated PCOS screening tool, and no model in this repository should be
> used for clinical decision-making.

## Dataset

| Property | Value |
|---|---|
| Respondents | 99 |
| Raw columns | 14 |
| Target | "Have you been medically diagnosed with PCOS?" |
| Class balance | 40 Yes / 59 No |

With 40 positive cases, the events-per-predictor ratio is about 4 at best,
well below the 10–20 conventionally required for stable model estimation.
Every performance figure is therefore reported with an uncertainty interval,
and model families are compared against each other, against a simple
logistic regression, and against a majority-class baseline.

**Data availability.** The survey workbook contains health information
from individual respondents and is not included in this repository.
<!-- Edit this line to reflect your data-sharing arrangement. -->

## What the pipeline does differently from the original script

The original analysis script had four issues that materially changed its
results. This version corrects each of them.

1. **Criterion leakage.** Menstrual cycle regularity, irregular periods,
   acne, and facial/body hair growth are components of the Rotterdam
   diagnostic criteria for PCOS. A model that includes them partly
   re-derives the diagnosis rather than predicting it. The pipeline
   therefore runs two pre-specified feature sets and reports both:
   - **SET_A**: lifestyle and demographic variables only
   - **SET_B**: SET_A plus self-reported symptoms

2. **Optimistic tuning bias.** The original script tuned hyperparameters
   with `RandomizedSearchCV` and then reported cross-validated performance
   on the same folds, which reuses test data for model selection and
   inflates ROC-AUC. This pipeline uses **nested cross-validation**, so
   tuning happens only within the training portion of each outer fold.

3. **Height and weight parsing.** The height column mixes formats such as
   `5'4`, `5 feet 3 inches`, `5.4` (meaning 5 ft 4 in), `158 cm`,
   `5.1 inches`, `5,2"`, `5/2 or 5/3`, and `4.4 ft`. The original parser
   returned `NaN` for the common bare-float entries, so BMI was missing for
   a large share of respondents and silently median-imputed. Weight
   contained strings such as `58 kg` and `40kg`, which forced the column to
   object dtype and produced `NaN` under `pd.to_numeric`. Both columns are
   now parsed explicitly.

4. **Discarded symptom information.** The original script collapsed the
   multi-select symptom field into a single count. This pipeline expands it
   into one binary indicator per symptom.

### Columns excluded as outcome leakage

The diagnostic test column and the test results column are excluded, as in
the original script. The test column is also missing for 31 respondents,
and that missingness is itself informative: a respondent who never had a
diagnostic test is unlikely to carry a diagnosis. Imputing it would bring
the outcome into the model.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── pcos_pipeline.py        # main analysis script
├── data/                   # place the survey workbook here (not tracked)
└── results/                # generated tables and figures
```
<!-- Update file names to match your repository. -->

## Usage

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python pcos_pipeline.py
```

Place the survey workbook in `data/` before running.

## Interpreting the results

- Report every metric with its fold-level spread or confidence interval, not
  as a single number.
- Compare SET_A and SET_B separately. Gains from SET_B reflect agreement
  between two self-reports (symptoms and diagnosis), not prediction.
- Expect overlapping performance across model families. If penalised
  logistic regression matches gradient boosting, that is a finding, not a
  failure.
