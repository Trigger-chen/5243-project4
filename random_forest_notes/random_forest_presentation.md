# Random Forest Regression — Presentation Script

---

## Slide 1: Goal

Our goal in this section is to predict IBM job salary using a Random Forest regression model.

Specifically, the target variable is the **log-transformed midpoint salary** — we call it `log_mid_salary`. We use the log scale for two reasons: salary distributions are right-skewed, and errors in log space correspond to proportional errors in dollars, which is a more meaningful measure of prediction quality for salaries that range from intern wages to senior engineer compensation.

The dataset is IBM job postings from `ibm_jobs.csv`, containing 469 rows with complete salary information.

---

## Slide 2: Data Split

Before touching any model, we reserved 20% of the data — 94 rows — as a completely held-out test set. The remaining 375 rows form the training set. This split was done once, with a fixed random seed of 42, and the test set was not used again until final evaluation.

This strict separation prevents any form of data leakage from influencing our reported performance.

---

## Note: Evaluation Strategy and Metrics

**Why both cross-validation and a train/test split?**

They answer different questions. Cross-validation (CV) is used during model development — it repeatedly rotates which portion of the training data is held out, giving a stable estimate of how well a given configuration generalizes. Because it averages over 5 folds, it reduces the luck-of-the-draw risk of any single split being unusually easy or hard. However, CV is also used to make decisions: we compare configurations and pick the best one. That decision process itself introduces a subtle bias — the winning configuration has, in effect, been selected to look good on these particular folds. The held-out test set is untouched by any of those decisions, so it gives an honest final estimate with no such bias. In short: CV guides development, the test set audits the final answer.

**Why RMSE?**

RMSE (Root Mean Squared Error) measures the typical size of a prediction error in the same units as the target. Squaring the errors before averaging penalizes large mistakes more than small ones — a prediction that's off by $40,000 hurts four times as much as one off by $20,000. Taking the square root brings the result back to interpretable units (log-salary, or dollars after back-transforming). It is the most common regression metric precisely because its units match the thing you are predicting.

**Why log scale for RMSE?**

Salary is modeled in log scale, so RMSE is also computed in log scale. A log-scale RMSE of 0.14 roughly means predictions are off by about ±15% of the actual salary. This is more meaningful than a dollar RMSE alone, because a $20,000 error means something very different for a $40,000 intern position versus a $200,000 senior role. Log-scale error treats those proportionally, which matches how we intuitively judge salary accuracy.

**Why R²?**

R² (coefficient of determination) is a normalized score between 0 and 1 that expresses how much of the salary variation the model explains. An R² of 0 means the model is no better than always predicting the mean salary; an R² of 1 means perfect predictions. Because it is unit-free and scale-free, it lets you compare models across different datasets or target transformations at a glance — something raw RMSE cannot do. We report both: RMSE tells you the practical error size in dollars, R² tells you how much signal the model captured.

---

## Slide 3: Preprocessing Pipeline

All preprocessing is embedded inside a scikit-learn `Pipeline` alongside the model, so every transformation is fit only on training data and then applied to validation and test folds consistently — there is no leakage through preprocessing either.

The preprocessor handles three different kinds of input features:

- **Numeric features (52 columns):** These include date-derived fields, location encoding, education ordinal levels, title and experience codes, seniority and role indicator flags, and skill flags. Each column receives median imputation for missing values, followed by standard scaling.

- **Categorical features (7 columns):** Fields like `state_province`, `area_of_work`, and `position_type`. These receive most-frequent imputation and are then one-hot encoded.

- **Text feature (1 field → up to 2,000 columns):** The `text_all` column combines the job title and experience description text. We apply TF-IDF vectorization with a vocabulary capped at 2,000 tokens — both unigrams and bigrams — with a minimum document frequency of 2 to filter out noise terms.

In total the model sees roughly 2,060 processed feature columns per row.

---

## Slide 4: Baseline Model

The baseline Random Forest uses 200 trees, `max_features = 0.33` — meaning each split considers one-third of all features, which is a standard starting point for regression — and out-of-bag scoring enabled.

We evaluated this using **5-fold cross-validation on the training set only**. The results:

| Metric | CV (validation folds) | Training folds |
|---|---|---|
| RMSE (log scale) | 0.1874 ± 0.0249 | 0.0724 |
| R² | 0.7371 ± 0.0343 | 0.9619 |

The large gap between training R² of 0.96 and CV R² of 0.74 is a clear sign of **overfitting**. The model memorizes the training data very well but generalizes imperfectly to unseen folds. This gap motivates hyperparameter tuning aimed at regularization.

---

## Slide 5: Hyperparameter Tuning

To address overfitting and find better generalization, we ran a **RandomizedSearchCV** over the following grid:

| Parameter | Options | What it controls |
|---|---|---|
| `n_estimators` | 200, 500, 1000 | Number of trees |
| `max_features` | 0.33, "sqrt", 0.2, 0.5 | Features considered at each split |
| `min_samples_leaf` | 1, 2, 5 | Minimum leaf size — higher values regularize deeper trees |
| `max_depth` | None, 20, 40 | Maximum tree depth — capping depth is another regularization lever |

We sampled 30 random combinations, each evaluated with 5-fold CV, for 150 total model fits. The best combination found was:

> `n_estimators=200`, `max_features=0.33`, `min_samples_leaf=1`, `max_depth=None`

This is identical to the baseline defaults. The tuning search found no improvement, indicating that the default parameter choice already sits near the optimum of the search space we explored.

---

## Slide 6: Test Set Evaluation

With tuning complete, the best model was refit on the full training set and evaluated on the held-out test set for the first and only time.

| Metric | Value |
|---|---|
| Baseline CV RMSE (log) | 0.1874 |
| Tuned CV RMSE (log) | 0.1874 |
| **Test RMSE (log scale)** | **0.1386** |
| **Test RMSE (USD)** | **$22,578** |
| **Test R²** | **0.8708** |

The test RMSE of 0.1386 is actually **lower** than the cross-validation RMSE of 0.1874 — the model performs better on the test set than on average CV folds. This does not indicate a problem with our evaluation setup. It most likely means the 20% test split happened to contain jobs that are somewhat easier to predict — for instance, more common position types with clearer salary signals. It is not evidence that CV overestimated error.

The test R² of 0.87 means our model explains about 87% of the variance in log salary on unseen data, with a typical prediction error of roughly $22,600 in dollar terms.

---

## Slide 7: Feature Importance

After fitting, we extracted **Mean Decrease in Impurity (MDI)** importances from the forest's trees. The top 20 features are shown in the bar chart.

The most important single feature is `cat__position_type_Professional` — the one-hot indicator for whether a posting is classified as a professional role. Close behind it is `cat__position_type_Internship`, which pulls salary strongly in the opposite direction. These two features alone capture a fundamental salary divide in IBM's job spectrum.

Beyond position type, education-related features appear prominently: `cat__preferred_education_Associate's Degree`, `num__preferred_edu_level`, and `num__required_edu_level` all rank in the top 20.

TF-IDF text features account for roughly half of the top 20 — words like `associate`, `apprentice`, `senior`, `oracle`, and `2026` contribute meaningful signal. This tells us that the raw text of a job description encodes salary information that structured fields alone do not fully capture.

Two temporal and contextual features also appear: `num__days_since_posted` and `cat__area_of_work_Infrastructure & Technology`, suggesting that recency and department assignment carry salary information.

---

## Slide 8: Summary and Limitations

To summarize:

- **Best test RMSE: $22,578** with an **R² of 0.87** — a reasonably strong predictive model on a relatively small dataset of 469 salaries.
- **Hyperparameter tuning found no improvement** over defaults. The default `max_features=0.33` and fully-grown trees are already near-optimal within the space we searched.
- **The model overfits** — a training R² of 0.96 versus a CV R² of 0.74 shows the trees are deeper than the signal in the data requires. One next step would be more aggressive regularization, for example, `min_samples_leaf` values above 5 or hard constraints on tree depth.
- **TF-IDF and position type jointly dominate** feature importance. This suggests that a simpler model using only position type, education level, and a handful of key text tokens might retain most of the predictive power with far less complexity.
- The dataset of 469 rows is small for a model with over 2,000 input dimensions. This high-dimensional, low-sample regime is inherently prone to overfitting and limits how far we can push accuracy without collecting more data.
