# IBM Job Posting Data Science Analysis

**GitHub Repository:** https://github.com/Trigger-chen/5243-project4

**Shiny Dashboard:** https://baixuanchen5243.shinyapps.io/5243-project4/

This repository contains the full pipeline, data, report, and deployment components for the final STAT5243 project.

## Project Overview

This project documents an end-to-end data science pipeline for predicting salary from publicly available IBM job postings. The dataset is obtained via web scraping IBM's careers website and subsequently cleaned, explored, and transformed through feature engineering. The processed data is then used to train and compare multiple supervised learning models: Random Forest, Ridge Regression, and Gradient Boosting.

The central research question is: Based on IBM's posted job listings, how accurately can we predict salary, and which features are the primary drivers of compensation?

## Repository Structure

```text
5243-project4/
│
├── Data/
│   ├── ibm_scraping.ipynb               # Selenium-based web scraper notebook
│   ├── ibm_jobs_raw.xls                 # Raw scraped dataset
│   └── ibm_jobs_final_processed.xls     # Final cleaned and processed dataset
│
├── Figures/                             # Saved figures used in report
│   ├── model_images/                    # Plots from supervised modeling section
│   ├── area_of_work_bar_plot.png
│   ├── areaofwork_positiontype_heatmap.png
│   ├── correlation_structure_salary_features.png
│   ├── feature_region_salary.png
│   ├── feature_seniority_salary.png
│   ├── jobpostingvolbypostypeovertime.png
│   ├── mid_salary_box_plot.png
│   ├── mid_salary_dist.png
│   ├── midsalary_by_areaofwork.png
│   ├── midsalary_by_positiontype.png
│   └── midsalary_by_required_education.png
│
├── Result/
│   ├── model result/                    # Supervised model outputs
│   │   ├── model_results.xls
│   │   ├── ridge_regression_results.xls
│   │   ├── rmse_comparison.png
│   │   └── r2_comparison.png
│   │
│   ├── unsupervised result/             # Clustering and outlier outputs
│   │   ├── ibm_jobs_with_unsupervised_results.csv
│   │   ├── cluster_profile_summary.csv
│   │   ├── cluster_categorical_profile.csv
│   │   ├── salary_summary_by_cluster.csv
│   │   └── outlier_postings.csv
│   │
│   └── model_selection_result.pdf       # Final model selection result
│
├── web/                                 # Shiny dashboard application
│   ├── app.R                            # Main Shiny app
│   ├── 5243-project4.Rproj              # RStudio project file
│   └── data/
│       └── ibm_jobs_final_processed.csv # Dataset used by Shiny app
│
├── project_4_new.ipynb                  # Main analysis notebook
├── report.md                            # Final written report
├── README.md                            # Repository documentation
└── 5243-project4.Rproj                  # Main RStudio project file
```

## Data

The raw dataset (`Data/ibm_jobs_raw.xls`) contains **478 job postings** scraped from IBM's public careers website, filtered to U.S. listings. Each posting includes 11 fields covering structural metadata, role attributes, education requirements, unstructured technical experience text, and posted salary endpoints.

| Column | Description |
|---|---|
| `job_title` | Job title as posted, such as "Senior Software Engineer". |
| `job_id` | Unique IBM-assigned posting identifier. |
| `date_posted` | Posting date in `DD-MMM-YYYY` format. |
| `state_province` | State(s) where the role is based; may contain multiple comma-separated states. |
| `area_of_work` | Business area, such as Software Engineering, Consulting, or Sales. |
| `min_salary` | Projected minimum annual salary in USD. |
| `max_salary` | Projected maximum annual salary in USD. |
| `position_type` | Role level, such as Professional, Entry Level, Internship, or Administration & Technician. |
| `required_education` | Minimum required education, such as Bachelor's Degree. |
| `preferred_education` | Preferred education level; often missing. |
| `preferred_technical_experience` | Free-text description of preferred technical skills; often missing. |

The cleaning pipeline produces the following sample sizes at each stage:

| Stage | n |
|---|---:|
| Raw scraped dataset | 478 |
| After deduplicating on `job_id` (7 duplicates removed) | 471 |
| After dropping rows missing core salary fields used in EDA | 470 |
| Final modeling feature matrix | 469 |

The analysis-ready dataset spans **August 14, 2025 to February 4, 2026**, with `mid_salary` ranging from approximately **$35K to $342K** and a median around **$134K**. The cleaned and feature-engineered version is saved as `Data/ibm_jobs_final_processed.xls`.

## Methodology

The pipeline is implemented across `Data/ibm_scraping.ipynb` for data acquisition, `project_4_new.ipynb` for cleaning, EDA, feature engineering, unsupervised learning, and modeling, and `Result/model_selection_result.pdf` for the final model comparison and selection summary.

The project follows these main stages:

1. **Data Acquisition.** Selenium with headless Chrome scrapes the IBM careers site, paginating through search-result pages and visiting each posting's detail page to extract structured fields.

2. **Data Cleaning and Handling Inconsistencies.** This includes type conversion, salary parsing, date parsing, formatting normalization, duplicate removal, and initial missing-value handling.

3. **Exploratory Data Analysis.** We conduct data quality checks, univariate distributions, bivariate and multivariate analysis, statistical tests, skill-keyword frequency analysis, time-trend analysis, and outlier checks.

4. **Data Preprocessing.** The final dataset is prepared using missing-value strategies, rare-category handling, categorical encoding, numerical scaling, and leakage prevention.

5. **Feature Engineering.** We create salary variables, date features, location features, ordinal education encodings, job-title seniority indicators, role-category indicators, and skill flags extracted from technical experience text.

6. **Unsupervised Learning.** We apply correlation analysis, PCA for dimensionality reduction, K-Means clustering with silhouette-based selection, and Isolation Forest outlier detection. Outputs are saved to `Result/unsupervised result/`.

7. **Supervised Learning.** Three models are trained with cross-validation and hyperparameter tuning on the log-transformed `mid_salary` target: Ridge Regression, Random Forest Regressor, and Gradient Boosting Regressor.

8. **Model Comparison and Selection.** Models are compared using RMSE, MAE, and R², along with qualitative considerations such as interpretability and robustness. The final selection summary is saved in `Result/model_selection_result.pdf`.

## Final Report

The final written report is available in `report.md`. It summarizes the full workflow, including data acquisition, EDA, feature engineering, preprocessing, unsupervised learning, supervised modeling, model comparison, dashboard development, challenges, and conclusions.

## Shiny Application

We produced an interactive Shiny dashboard to allow users to explore the IBM job posting dataset and view key outputs from EDA, feature engineering, unsupervised learning, and supervised modeling.

The deployed dashboard is available here:

```text
https://baixuanchen5243.shinyapps.io/5243-project4/
```

The Shiny application source code is located in:

```text
web/app.R
```

## How to Run

### Requirements

- Python 3.10+; the analysis was developed using Python 3.12.
- R and RStudio for running the Shiny dashboard.
- For scraping only: Google Chrome and a compatible ChromeDriver.

### Python Dependencies

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scipy scikit-learn selenium jupyter openpyxl
```

Note: `scikit-learn >= 1.4` is recommended for use of `root_mean_squared_error`.

### R Dependencies for Shiny App

Install the required R packages:

```r
install.packages(c(
  "shiny",
  "dplyr",
  "ggplot2",
  "plotly",
  "DT",
  "readr",
  "tidyr"
))
```

### Reproducing the Pipeline

1. **Optional: Re-scrape the data.**

   Because the IBM careers site changes over time, re-running the scraper may produce a different dataset from the snapshot provided in this repository. To re-scrape the data, open and run:

   ```text
   Data/ibm_scraping.ipynb
   ```

   This notebook regenerates the raw scraped dataset:

   ```text
   Data/ibm_jobs_raw.xls
   ```

   Selenium and ChromeDriver are required for this step. To reproduce the submitted results exactly, skip re-scraping and use the provided data snapshot.

2. **Run the main analysis notebook.**

   Open and run all cells in:

   ```text
   project_4_new.ipynb
   ```

   This notebook reads the raw dataset, performs cleaning, EDA, feature engineering, unsupervised learning, and modeling. It produces the cleaned/processed dataset, saved figures, and model/unsupervised outputs.

   Main outputs include:

   ```text
   Data/ibm_jobs_final_processed.xls
   Figures/
   Result/unsupervised result/
   Result/model result/
   ```

3. **Review final model comparison and selection.**

   The final model evaluation and selection summary is saved in:

   ```text
   Result/model_selection_result.pdf
   ```

   The report also summarizes the final model comparison using RMSE, MAE, and R².

4. **Run the Shiny app locally.**

   Open RStudio from the project folder, then run:

   ```r
   shiny::runApp("web/app.R")
   ```

   The Shiny app uses the processed dataset stored at:

   ```text
   web/data/ibm_jobs_final_processed.csv
   ```

   A deployed version of the dashboard is also available here:

   ```text
   https://baixuanchen5243.shinyapps.io/5243-project4/
   ```

## Notes on Data Files

The canonical processed dataset for the main analysis is stored in:

```text
Data/ibm_jobs_final_processed.xls
```

A separate CSV copy is stored in:

```text
web/data/ibm_jobs_final_processed.csv
```

This copy is included only to support the Shiny application.

## Notes

- **The scraped data is a snapshot, not a live feed.** The dataset represents IBM open postings at the moment of scraping. Re-scraping the website may produce a different snapshot.
- **Sample size is modest.** The final modeling dataset contains 469 postings, so model performance should be interpreted with this limitation in mind.
- **Salary leakage was avoided.** Salary-derived variables were used for EDA and interpretation but removed from the supervised model predictor matrix.
- **Shiny app data copy.** The `web/data/` folder contains a CSV copy of the processed dataset only so the Shiny dashboard can run independently.