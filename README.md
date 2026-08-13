PCOS PREDICTION FROM SURVEY DATA — REFINED ANALYSIS PIPELINE

Dataset characteristics (verified from the supplied workbook): n = 99 respondents, 14 raw columns
Target: "Have you been medically diagnosed with PCOS?"  ->  40 Yes / 59 No
#
# READ THIS BEFORE INTERPRETING ANY OUTPUT
# ----------------------------------------
# With 99 respondents and 40 positive cases, this dataset supports an
# exploratory, descriptive analysis. It does not support a claim that a
# tuned gradient boosting model is a validated PCOS screening tool. The
# events-per-predictor ratio is approximately 4 at best, which is well below
# the ~10-20 conventionally required for stable model estimation. Every
# performance number below must be reported with an uncertainty interval,
# and the honest conclusion is likely to be that the model families are
# statistically indistinguishable from one another and possibly from a
# simple logistic regression on two or three variables.
#
# The original script contained four issues that materially change results:
#
#   1. CRITERION LEAKAGE. Menstrual cycle regularity, irregular periods,
#      acne, and facial/body hair growth are not neutral predictors. They
#      are components of the Rotterdam diagnostic criteria for PCOS. A model
#      containing them is partially re-deriving the diagnosis rather than
#      predicting it. This script therefore runs two pre-specified feature
#      sets: SET_A (lifestyle and demographic only) and SET_B (SET_A plus
#      self-reported symptoms), and reports both.
#
#   2. OPTIMISTIC TUNING BIAS. The original script selected hyperparameters
#      by RandomizedSearchCV over folds, then reported cross-validated
#      performance using those same folds. That reuses the test data for
#      model selection and inflates ROC-AUC. This script uses nested
#      cross-validation: tuning happens inside the training portion only.
#
#   3. HEIGHT AND WEIGHT PARSING FAILURE. The height column contains
#      "5'4", "5 feet 3 inches", 5.4 (meaning five feet four inches),
#      "158 cm", "5.1 inches", "5,2\"", "5/2 or 5/3", and "4.4 ft". The
#      original height_to_inches function returns NaN for the bare float
#      entries, which are common, so BMI was missing for a large share of
#      respondents and was then silently median-imputed. Weight also
#      contains strings such as "58 kg" and "40kg", which made the whole
#      column object dtype and produced NaN under pd.to_numeric.
#
#   4. SYMPTOM INFORMATION DISCARDED. Collapsing a multi-select symptom
#      field to a single count throws away which symptoms were reported.
#      This script expands it into binary indicators.
#
# Also excluded as outcome leakage, as in the original script: the
# diagnostic test column and the results column. Note additionally that the
# test column is missing for 31 respondents, and that missingness is itself
# informative (a respondent who underwent no diagnostic test is unlikely to
# carry a diagnosis). Imputing it would import the outcome into the model.
#
