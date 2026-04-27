# Random Forest Regression — Worksheet
**Target:** `log_mid_salary` (log-transformed midpoint salary)  
**Dataset:** IBM job postings (`ibm_jobs.csv`)

---

## Step 1: Train / Test Split

The full dataset is split into a training set (80%) and a held-out test set (20%).  
The test set is **not touched** until final evaluation.

| | Value |
|---|---|
| Total usable rows (non-missing salary) | ________ |
| Training rows | ________ |
| Test rows | ________ |
| Random seed | 42 |

---

## Step 2: Preprocessing

A `ColumnTransformer` is applied inside the pipeline so that all fitting happens on training data only.  
The preprocessor has three branches:

| Branch | Columns | Steps |
|---|---|---|
| Numeric (52 features) | Date, location, education ordinal, title, experience, seniority/role flags, skill flags | Median imputation → StandardScaler |
| Categorical (7 features) | state_province, primary_state, primary_region, area_of_work, position_type, required_education, preferred_education | Most-frequent imputation → OneHotEncoder |
| Text (1 field → up to 2000 features) | `text_all` (job title + experience text combined) | TF-IDF (max 2000 features, unigrams + bigrams, min_df=2) |

Total processed feature columns: ________

---

## Step 3: Baseline Cross-Validation (Default Params)

**Params used:** `n_estimators=200`, `max_features=0.33`, `random_state=42`

5-fold CV on training set only.

| Metric | Mean | Std |
|---|---|---|
| CV RMSE (log scale) | ________ | ________ |
| CV R² | ________ | ________ |
| Train RMSE (log scale) | ________ | ________ |
| Train R² | ________ | ________ |

**Interpretation:**  
Large gap between train and CV scores → ________ (overfit / underfit / reasonable)

---

## Step 4: Hyperparameter Tuning (RandomizedSearchCV)

**Search strategy:** 30 random combinations, 5-fold CV, scored on RMSE  
**Parameter grid searched:**

| Parameter | Values tried | Meaning |
|---|---|---|
| `n_estimators` | 200, 500, 1000 | Number of trees in the forest |
| `max_features` | 0.33, "sqrt", 0.2, 0.5 | m_try — features considered at each split |
| `min_samples_leaf` | 1, 2, 5 | Minimum samples required at a leaf node |
| `max_depth` | None, 20, 40 | Maximum depth of each tree |

**Best parameters found:**

| Parameter | Best value |
|---|---|
| `n_estimators` | ________ |
| `max_features` | ________ |
| `min_samples_leaf` | ________ |
| `max_depth` | ________ |

Best CV RMSE (log scale) from search: ________

---

## Step 5: Final Evaluation on Test Set

The best model (refit on all of `X_train`) is evaluated on the held-out test set.

| Metric | Value |
|---|---|
| Baseline CV RMSE (log) | ________ |
| Tuned CV RMSE (log) | ________ |
| **Tuned Test RMSE (log)** | ________ |
| **Tuned Test RMSE (USD)** | $________ |
| **Tuned Test R²** | ________ |

**Did tuning help over baseline?** ________ (yes / marginal / no)  
**Is test RMSE close to tuned CV RMSE?** ________ (if not, possible CV overfit)

---

## Step 6: Feature Importance (Top 20, MDI)

Mean Decrease in Impurity — higher = more influential at splits.  
Note: TF-IDF features often dominate raw importance scores.

| Rank | Feature name | Importance score |
|---|---|---|
| 1 | ________ | ________ |
| 2 | ________ | ________ |
| 3 | ________ | ________ |
| 4 | ________ | ________ |
| 5 | ________ | ________ |
| 6 | ________ | ________ |
| 7 | ________ | ________ |
| 8 | ________ | ________ |
| 9 | ________ | ________ |
| 10 | ________ | ________ |

**Top engineered feature (excluding TF-IDF):** ________  
**Dominant feature group** (TF-IDF / seniority flags / area_of_work / other): ________

---

## Summary

| Question | Answer |
|---|---|
| Best RMSE achieved (USD) | $________ |
| Best R² on test set | ________ |
| Most important feature | ________ |
| Did TF-IDF features dominate? | ________ |
| Main limitation | ________ |
