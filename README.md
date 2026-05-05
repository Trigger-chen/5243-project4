# IBM Job Posting Data Science Analysis

## Project Overview
This project documents an end-to-end data science pipeline for predicting salary from publicly available IBM job postings. The dataset is acquired by web-scraping IBM's careers site, cleaned and prepared, explored through descriptive and inferential analysis, transformed via feature engineering, and used to train and compare three supervised learning models: Random Forest, Ridge Regression, and Gradient Boosting.

The central research question is: Based on IBM's posted job listings, how accurately can we predict salary, and which features are the primary drivers of compensation?

## Repository Structure 
```text
5243-project4/
|
├── data/
│   ├── ibm_scraping.ipynb               # Selenium-based web scraper notebook
│   ├── ibm_jobs_raw.csv                 # Raw scraped dataset (478 postings × 11 columns)
│   └── ibm_jobs_final_processed.csv     # Cleaned + feature-engineered dataset
│
├── project_4_new.ipynb                  # Main analysis notebook (cleaning → EDA → modeling)
├── model_evaluation_and_selection.ipynb # Final model comparison and selection
│
├── Figures/                             # Saved EDA and modeling figures (.png)
│   └── model_images/                    # Plots from supervised modeling section
│
├── result/
│   ├── ridge_regression_results.xls
│   └── unsupervised result/
│       ├── ibm_jobs_with_unsupervised_results.csv  # Dataset with cluster + outlier labels
│       ├── cluster_profile_summary.csv             # Numeric K-means cluster profiles
│       ├── cluster_categorical_profile.csv         # Categorical K-means cluster profiles
│       ├── salary_summary_by_cluster.csv           # Salary stats per cluster
│       └── outlier_postings.csv                    # Postings flagged by Isolation Forest
│
├── 5243_Final_Project__model_evaluation_and_selection_.pdf  # Model evaluation report (PDF)
├── report.md                            # Final project report (Markdown source)
└── README.md
```

## Data

The raw dataset (`data/ibm_jobs_raw.csv`) contains **478 job postings** scraped from IBM's public careers website, filtered to U.S. listings. Each posting includes 11 fields covering structural metadata, role attributes, education requirements, unstructured technical experience text, and posted salary endpoints.

| Column | Description |
|---|---|
| `job_title` | Job title as posted (e.g., "Senior Software Engineer") |
| `job_id` | Unique IBM-assigned posting identifier |
| `date_posted` | Posting date in `DD-MMM-YYYY` format |
| `state_province` | State(s) where the role is based; may contain multiple comma-separated states |
| `area_of_work` | Business area (e.g., Software Engineering, Consulting, Sales) |
| `min_salary` | Projected minimum annual salary (USD) |
| `max_salary` | Projected maximum annual salary (USD) |
| `position_type` | Role level: Professional, Entry Level, Internship, or Administration & Technician |
| `required_education` | Minimum required education (e.g., Bachelor's Degree) |
| `preferred_education` | Preferred education level (often missing) |
| `preferred_technical_experience` | Free-text description of preferred technical skills (often missing) |

The cleaning pipeline produces the following sample sizes at each stage:

| Stage | n |
|---|---|
| Raw scraped dataset | 478 |
| After deduplicating on `job_id` (7 duplicates removed) | 471 |
| After dropping rows missing core salary fields (used in EDA) | 470 |
| Final modeling feature matrix (Section 6.8) | 469 |

The analysis-ready dataset spans **August 14, 2025 to February 4, 2026**, with `mid_salary` (the midpoint of the posted band) ranging from approximately $35K to $342K and a median around $134K. The cleaned and feature-engineered version is saved as `data/ibm_jobs_final_processed.csv`.

## Methodology

The pipeline is implemented across `data/ibm_scraping.ipynb` (data acquisition), `project_4_new.ipynb` (cleaning, EDA, feature engineering, and initial modeling), and `model_evaluation_and_selection.ipynb` (final model comparison), with the following stages:

1. **Data Acquisition.** Selenium with headless Chrome scrapes the IBM careers site, paginating through 17 search-result pages and visiting each posting's detail page to extract 11 structured fields.

2. **Data Cleaning and Handling Inconsistencies.** Type conversion (string-to-numeric salary, date parsing), formatting normalization (whitespace, casing), duplicate removal (7 duplicates → 471 unique postings), and initial missing-value handling.

3. **Exploratory Data Analysis.** Data quality checks, univariate distributions, bivariate and multivariate analysis with statistical tests (Kruskal–Wallis on salary vs. area of work, position type, and required education; chi-square test of association between area and position type), skill-keyword frequency analysis, time-trend analysis, and outlier checks.

4. **Data Preprocessing.** Final dataset selection, formal missing-value strategy, rare-category pooling, encoding for categorical variables (one-hot for nominal, ordinal for education), and numerical scaling.

5. **Feature Engineering.** Engineered salary variables, date features (`days_since_posted`, `posted_month`), location features (multi-state indicator, primary region), ordinal education encoding, job-title parsing for seniority signals (`is_senior`, `is_manager`, `is_director_plus`, etc.), and 18 binary skill indicator flags extracted from the unstructured experience text.

6. **Unsupervised Learning.** Correlation analysis, PCA for dimensionality reduction, K-means clustering with silhouette-based k selection, and Isolation Forest outlier detection on the engineered feature space. Outputs are saved to `result/unsupervised result/`.

7. **Supervised Learning.** Three models trained with cross-validation and hyperparameter tuning on the log-transformed `mid_salary` target: Ridge Regression (linear baseline with TF-IDF text features), Random Forest Regressor, and Gradient Boosting Regressor.

8. **Model Comparison and Selection.** Quantitative comparison via RMSE, MAE, and R², combined with qualitative considerations (interpretability, robustness) to select a final model. See `model_evaluation_and_selection.ipynb`.

## How to Run

### Requirements

- Python 3.10+ (developed on 3.12)
- For scraping only: Google Chrome and a compatible ChromeDriver

### Dependencies

```bash
pip install pandas numpy matplotlib scipy scikit-learn selenium
```

### Reproducing the Pipeline

1. **(Optional) Re-scrape the data.** Because the IBM careers site changes daily, the scraper will produce a different dataset than the one provided here. Open and run `data/ibm_scraping.ipynb` to regenerate `data/ibm_jobs_raw.csv`. Selenium and a Chrome WebDriver are required. Skip this step to use the provided snapshot.

2. **Run the main analysis.** Open `project_4_new.ipynb` and run all cells in order. The notebook reads `data/ibm_jobs_raw.csv` and produces inline EDA figures and statistical-test outputs, the cleaned/processed dataset (`data/ibm_jobs_final_processed.csv`), saved figures (in `Figures/`), and the unsupervised learning outputs (in `result/unsupervised result/`).

3. **Run final model comparison.** Open `model_evaluation_and_selection.ipynb` to reproduce the final supervised-model evaluation and selection.

## Notes

- **The web scraped data is a snapshot and not a live feed.** The dataset represents IBM's open postings at the moment of scraping (postings dated August 2025 through February 2026). Re-scraping the website will produce a different snapshot of the data. 
- **End-loaded posting volume.** Roughly 55% of postings in the dataset are dated January 2026, with earlier months being a lot sparser. This implies that older postings tend to be filled and removed before the web scrape runs. 
- **Sample size.** With ~470 postings after cleaning, the dataset is modest. The pipeline is designed to be reproducible on larger snapshots if more data is collected over time.


