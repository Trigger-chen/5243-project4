# Random Forest Regression — Worksheet
**Target:** `log_mid_salary` (log-transformed midpoint salary)  
**Dataset:** IBM job postings (`ibm_jobs.csv`)

---

## Step 1: Train / Test Split

The full dataset is split into a training set (80%) and a held-out test set (20%).  
The test set is **not touched** until final evaluation.

| | Value |
|---|---|
| Total usable rows (non-missing salary) | 469 |
| Training rows | 375 |
| Test rows | 94 |
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
| CV RMSE (log scale) | 0.1874 | 0.0249 |
| CV R² | 0.7371 | 0.0343 |
| Train RMSE (log scale) | 0.0724 | — |
| Train R² | 0.9619 | — |

**Interpretation:**  
Large gap between train and CV scores → overfit (overfit / underfit / reasonable)

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
| `n_estimators` | 200 |
| `max_features` | 0.33 |
| `min_samples_leaf` | 1 |
| `max_depth` | None |

Best CV RMSE (log scale) from search: 0.1874

---

## Step 5: Final Evaluation on Test Set

The best model (refit on all of `X_train`) is evaluated on the held-out test set.

| Metric | Value |
|---|---|
| Baseline CV RMSE (log) | 0.1874 |
| Tuned CV RMSE (log) | 0.1874 |
| **Tuned Test RMSE (log)** | 0.1386 |
| **Tuned Test RMSE (USD)** | $22,578 |
| **Tuned Test R²** | 0.8708 |

**Did tuning help over baseline?** no (yes / marginal / no)  
**Is test RMSE close to tuned CV RMSE?** Test RMSE (0.1386) is actually better than CV RMSE (0.1874) — the test set was likely easier than the average CV fold, not a sign of CV overfit

---

## Step 6: Feature Importance (Top 20, MDI)

Mean Decrease in Impurity — higher = more influential at splits.  
Note: TF-IDF features often dominate raw importance scores.

Top 20 features:
cat__position_type_Professional
cat__position_type_Internship
cat__preferred_education_Associate's Degree/College Diploma
txt__associate
txt__2026
cat__position_type_Entry Level
txt__team based
cat__position_type_Administration & Technician
txt__apprentice
cat__required_education_High School Diploma/GED
txt__exposure
cat__area_of_work_Infrastructure & Technology
num__preferred_edu_level
txt__oracle
txt__representative
num__required_edu_level
txt__senior
num__days_since_posted
txt__fast
txt__academic

**Dominant feature group** (TF-IDF / seniority flags / area_of_work / other): TF-IDF

---

## Summary

| Question | Answer |
|---|---|
| Best RMSE achieved (USD) | $22,578 |
| Best R² on test set | 0.8708 |
| Most important feature | cat__position_type_Professional |
| Did TF-IDF features dominate? | Partially — 10/20 top features are TF-IDF, but the single top feature is categorical (position_type) |
| Main limitation | Overfitting (CV R² = 0.74 vs. likely near-perfect train R²); tuning found no improvement over defaults |
