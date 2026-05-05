library(shiny)
library(dplyr)
library(ggplot2)
library(plotly)
library(DT)
library(readr)
library(tidyr)

# ============================================================
# 1. Load Data
# ============================================================

df <- read_csv("data/ibm_jobs_final_processed.csv", show_col_types = FALSE)

if ("cluster" %in% names(df)) {
  df$cluster <- as.factor(df$cluster)
}

if ("is_outlier" %in% names(df)) {
  df$is_outlier <- as.logical(df$is_outlier)
}

if (!("log_mid_salary" %in% names(df)) && "mid_salary" %in% names(df)) {
  df$log_mid_salary <- log(df$mid_salary)
}

if (!("salary_range" %in% names(df)) && all(c("min_salary", "max_salary") %in% names(df))) {
  df$salary_range <- df$max_salary - df$min_salary
}

# ============================================================
# 2. Helper Functions
# ============================================================

get_choices <- function(data, col) {
  if (col %in% names(data)) {
    c("All", sort(unique(na.omit(as.character(data[[col]])))))
  } else {
    c("All")
  }
}

has_cols <- function(data, cols) {
  all(cols %in% names(data))
}

get_salary_values <- function(data, col = "mid_salary") {
  if (!(col %in% names(data))) {
    return(numeric(0))
  }
  
  values <- data[[col]]
  values <- values[!is.na(values)]
  
  if (length(values) == 0) {
    return(numeric(0))
  }
  
  values
}

format_dollar <- function(x) {
  paste0("$", format(round(x, 0), big.mark = ","))
}

plotly_clean <- function(p) {
  ggplotly(p, tooltip = "text") %>%
    config(displayModeBar = FALSE) %>%
    layout(
      margin = list(l = 80, r = 30, t = 60, b = 60),
      title = list(font = list(size = 16))
    )
}

empty_plot <- function(message) {
  p <- ggplot() +
    annotate("text", x = 0, y = 0, label = message, size = 4) +
    theme_void()
  
  plotly_clean(p)
}

base_theme <- function() {
  theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(size = 13, face = "bold", hjust = 0),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8),
      plot.margin = margin(t = 10, r = 20, b = 20, l = 20)
    )
}

make_count_plot <- function(data, col_name, plot_title, y_label) {
  if (!(col_name %in% names(data))) {
    return(empty_plot(paste(col_name, "is not available.")))
  }
  
  plot_df <- data %>%
    filter(!is.na(.data[[col_name]])) %>%
    count(.data[[col_name]], sort = TRUE) %>%
    head(10) %>%
    arrange(n)
  
  if (nrow(plot_df) == 0) {
    return(empty_plot("No data available for the selected filters."))
  }
  
  names(plot_df)[1] <- "category"
  plot_df$category <- factor(plot_df$category, levels = plot_df$category)
  
  p <- ggplot(
    plot_df,
    aes(
      x = n,
      y = category,
      text = paste0(y_label, ": ", category, "<br>Count: ", n)
    )
  ) +
    geom_col(fill = "steelblue") +
    labs(
      title = plot_title,
      x = "Count",
      y = y_label
    ) +
    base_theme()
  
  plotly_clean(p)
}

make_hist_plot <- function(data, col_name, plot_title, x_label, bins = 30) {
  if (!(col_name %in% names(data))) {
    return(empty_plot(paste(col_name, "is not available.")))
  }
  
  plot_df <- data %>%
    filter(!is.na(.data[[col_name]]))
  
  if (nrow(plot_df) == 0) {
    return(empty_plot("No data available for the selected filters."))
  }
  
  p <- ggplot(
    plot_df,
    aes(
      x = .data[[col_name]],
      text = paste0(x_label, ": ", round(.data[[col_name]], 2))
    )
  ) +
    geom_histogram(bins = bins, fill = "steelblue", color = "white") +
    labs(
      title = plot_title,
      x = x_label,
      y = "Count"
    ) +
    base_theme()
  
  plotly_clean(p)
}

make_horizontal_box_plot <- function(data, category_col, salary_col, plot_title, category_label, salary_label) {
  if (!has_cols(data, c(category_col, salary_col))) {
    return(empty_plot("Required columns are not available."))
  }
  
  plot_df <- data %>%
    filter(!is.na(.data[[category_col]]), !is.na(.data[[salary_col]]))
  
  if (nrow(plot_df) == 0) {
    return(empty_plot("No data available for the selected filters."))
  }
  
  order_df <- plot_df %>%
    group_by(.data[[category_col]]) %>%
    summarize(median_salary = median(.data[[salary_col]], na.rm = TRUE), .groups = "drop") %>%
    arrange(median_salary)
  
  levels_order <- order_df[[category_col]]
  plot_df[[category_col]] <- factor(plot_df[[category_col]], levels = levels_order)
  
  p <- ggplot(
    plot_df,
    aes(
      x = .data[[salary_col]],
      y = .data[[category_col]],
      text = paste0(
        category_label, ": ", .data[[category_col]],
        "<br>", salary_label, ": $", round(.data[[salary_col]], 0)
      )
    )
  ) +
    geom_boxplot(fill = "lightblue") +
    labs(
      title = plot_title,
      x = salary_label,
      y = category_label
    ) +
    base_theme()
  
  plotly_clean(p)
}

make_median_bar_plot <- function(data, group_col, salary_col, plot_title, group_label, salary_label) {
  if (!has_cols(data, c(group_col, salary_col))) {
    return(empty_plot("Required columns are not available."))
  }
  
  plot_df <- data %>%
    filter(!is.na(.data[[group_col]]), !is.na(.data[[salary_col]])) %>%
    group_by(.data[[group_col]]) %>%
    summarize(
      median_salary = median(.data[[salary_col]], na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    arrange(median_salary)
  
  if (nrow(plot_df) == 0) {
    return(empty_plot("No data available for the selected filters."))
  }
  
  names(plot_df)[1] <- "group"
  plot_df$group <- factor(plot_df$group, levels = plot_df$group)
  
  p <- ggplot(
    plot_df,
    aes(
      x = median_salary,
      y = group,
      text = paste0(
        group_label, ": ", group,
        "<br>Median Salary: $", round(median_salary, 0),
        "<br>Count: ", n
      )
    )
  ) +
    geom_col(fill = "steelblue") +
    labs(
      title = plot_title,
      x = salary_label,
      y = group_label
    ) +
    base_theme()
  
  plotly_clean(p)
}

# Safe correlation function
safe_cor <- function(x, y) {
  complete_idx <- complete.cases(x, y)
  
  if (sum(complete_idx) < 3) {
    return(NA_real_)
  }
  
  x2 <- x[complete_idx]
  y2 <- y[complete_idx]
  
  if (sd(x2, na.rm = TRUE) == 0 || sd(y2, na.rm = TRUE) == 0) {
    return(NA_real_)
  }
  
  out <- suppressWarnings(cor(x2, y2))
  
  if (!is.finite(out)) {
    return(NA_real_)
  }
  
  out
}

# ============================================================
# 3. UI
# ============================================================

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      .container-fluid {
        max-width: 100%;
      }
      .tab-content {
        padding-top: 12px;
      }
      .plotly {
        width: 100% !important;
      }
      h3 {
        margin-top: 18px;
        margin-bottom: 16px;
      }
      h4 {
        margin-top: 20px;
        margin-bottom: 10px;
      }
      .model-note {
        font-size: 15px;
        line-height: 1.6;
        max-width: 1000px;
      }
    "))
  ),
  
  titlePanel("IBM Job Salary Analysis Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      selectInput(
        "area_filter",
        "Area of Work",
        choices = get_choices(df, "area_of_work"),
        selected = "All"
      ),
      
      selectInput(
        "position_filter",
        "Position Type",
        choices = get_choices(df, "position_type"),
        selected = "All"
      ),
      
      selectInput(
        "region_filter",
        "Primary Region",
        choices = get_choices(df, "primary_region"),
        selected = "All"
      ),
      
      hr(),
      p("Interactive dashboard for the IBM job salary final project."),
      p("Use the filters above to explore how salary patterns change across work areas, position types, and regions.")
    ),
    
    mainPanel(
      width = 9,
      
      tabsetPanel(
        tabPanel(
          "Overview",
          br(),
          h3("Dataset Overview"),
          fluidRow(
            column(3, strong("Job Postings"), textOutput("n_jobs")),
            column(3, strong("Average Salary"), textOutput("avg_salary")),
            column(3, strong("Median Salary"), textOutput("median_salary")),
            column(3, strong("Clusters"), textOutput("n_clusters"))
          ),
          br(),
          fluidRow(
            column(3, strong("Number of Features"), textOutput("n_features")),
            column(3, strong("Salary Min"), textOutput("salary_min")),
            column(3, strong("Salary Max"), textOutput("salary_max")),
            column(3, strong("Outliers"), textOutput("n_outliers"))
          ),
          br(),
          h4("Processed Dataset Preview"),
          DTOutput("data_preview")
        ),
        
        tabPanel(
          "EDA",
          br(),
          h3("Exploratory Data Analysis"),
          
          h4("Data Quality Summary"),
          DTOutput("missing_table"),
          br(),
          
          h4("Categorical Variable Distributions"),
          plotlyOutput("position_count_plot", height = "340px", width = "100%"),
          br(),
          plotlyOutput("area_count_plot", height = "400px", width = "100%"),
          br(),
          plotlyOutput("region_count_plot", height = "340px", width = "100%"),
          br(),
          plotlyOutput("education_count_plot", height = "400px", width = "100%"),
          br(),
          
          h4("Salary Distribution"),
          plotlyOutput("salary_hist", height = "340px", width = "100%"),
          br(),
          plotlyOutput("log_salary_hist", height = "340px", width = "100%"),
          br(),
          
          h4("Salary Range"),
          plotlyOutput("salary_range_hist", height = "340px", width = "100%"),
          br(),
          
          h4("Correlations with Log Salary"),
          plotlyOutput("corr_salary_plot", height = "440px", width = "100%")
        ),
        
        tabPanel(
          "Salary Exploration",
          br(),
          h3("Salary Exploration"),
          plotlyOutput("salary_by_area", height = "400px", width = "100%"),
          br(),
          plotlyOutput("salary_by_position", height = "360px", width = "100%"),
          br(),
          plotlyOutput("salary_by_region", height = "340px", width = "100%"),
          br(),
          plotlyOutput("salary_by_required_edu", height = "440px", width = "100%"),
          br(),
          plotlyOutput("salary_by_preferred_edu", height = "440px", width = "100%")
        ),
        
        tabPanel(
          "Feature Engineering",
          br(),
          h3("Feature Engineering Summary"),
          p("This section summarizes important engineered features used in the final processed dataset."),
          
          h4("Available Engineered Features"),
          DTOutput("engineered_feature_table"),
          br(),
          
          h4("Skills Mentioned Distribution"),
          plotlyOutput("skills_hist", height = "340px", width = "100%"),
          br(),
          
          h4("Experience Requirement Distribution"),
          plotlyOutput("exp_min_hist", height = "340px", width = "100%"),
          br(),
          plotlyOutput("exp_max_hist", height = "340px", width = "100%"),
          br(),
          
          h4("Role Indicator Summary"),
          DTOutput("role_summary_table")
        ),
        
        tabPanel(
          "Unsupervised Learning",
          br(),
          h3("Unsupervised Learning Results"),
          p("This section visualizes PCA, K-Means clustering, and outlier detection results if these columns are available in the final dataset."),
          
          plotlyOutput("pca_cluster_plot", height = "440px", width = "100%"),
          br(),
          plotlyOutput("salary_by_cluster", height = "340px", width = "100%"),
          br(),
          
          h4("Cluster Summary"),
          DTOutput("cluster_summary_table"),
          br(),
          
          h4("Outlier Job Postings"),
          DTOutput("outlier_table")
        ),
        
        tabPanel(
          "Modeling",
          br(),
          h3("Supervised Modeling Results"),
          p("The supervised models were evaluated using a unified testing approach. The target variable is log-transformed mid salary."),
          
          h4("Detailed Log-scale Model Metrics"),
          DTOutput("model_log_table"),
          br(),
          
          h4("Dollar-scale Error Comparison"),
          DTOutput("model_usd_table"),
          br(),
          
          plotlyOutput("rmse_plot", height = "340px", width = "100%"),
          br(),
          plotlyOutput("mae_plot", height = "340px", width = "100%"),
          br(),
          plotlyOutput("r2_plot", height = "340px", width = "100%"),
          br(),
          
          h4("Final Model Selection"),
          htmlOutput("final_model_text")
        )
      )
    )
  )
)

# ============================================================
# 4. Server
# ============================================================

server <- function(input, output, session) {
  
  filtered_df <- reactive({
    temp <- df
    
    if ("area_of_work" %in% names(temp) && input$area_filter != "All") {
      temp <- temp %>% filter(area_of_work == input$area_filter)
    }
    
    if ("position_type" %in% names(temp) && input$position_filter != "All") {
      temp <- temp %>% filter(position_type == input$position_filter)
    }
    
    if ("primary_region" %in% names(temp) && input$region_filter != "All") {
      temp <- temp %>% filter(primary_region == input$region_filter)
    }
    
    temp
  })
  
  # ------------------------------------------------------------
  # Overview
  # ------------------------------------------------------------
  
  output$n_jobs <- renderText({
    format(nrow(filtered_df()), big.mark = ",")
  })
  
  output$n_features <- renderText({
    format(ncol(filtered_df()), big.mark = ",")
  })
  
  output$avg_salary <- renderText({
    salary_values <- get_salary_values(filtered_df(), "mid_salary")
    
    if (length(salary_values) == 0) {
      return("N/A")
    }
    
    format_dollar(mean(salary_values))
  })
  
  output$median_salary <- renderText({
    salary_values <- get_salary_values(filtered_df(), "mid_salary")
    
    if (length(salary_values) == 0) {
      return("N/A")
    }
    
    format_dollar(median(salary_values))
  })
  
  output$salary_min <- renderText({
    salary_values <- get_salary_values(filtered_df(), "mid_salary")
    
    if (length(salary_values) == 0) {
      return("N/A")
    }
    
    format_dollar(min(salary_values))
  })
  
  output$salary_max <- renderText({
    salary_values <- get_salary_values(filtered_df(), "mid_salary")
    
    if (length(salary_values) == 0) {
      return("N/A")
    }
    
    format_dollar(max(salary_values))
  })
  
  output$n_clusters <- renderText({
    if ("cluster" %in% names(filtered_df())) {
      length(unique(na.omit(filtered_df()$cluster)))
    } else {
      "N/A"
    }
  })
  
  output$n_outliers <- renderText({
    if ("is_outlier" %in% names(filtered_df())) {
      sum(filtered_df()$is_outlier == TRUE, na.rm = TRUE)
    } else {
      "N/A"
    }
  })
  
  output$data_preview <- renderDT({
    show_cols <- c(
      "job_title",
      "area_of_work",
      "position_type",
      "primary_region",
      "mid_salary",
      "log_mid_salary",
      "required_education",
      "preferred_education",
      "n_skills_mentioned",
      "exp_years_min",
      "exp_years_max",
      "cluster",
      "is_outlier"
    )
    
    show_cols <- show_cols[show_cols %in% names(filtered_df())]
    
    datatable(
      filtered_df()[, show_cols],
      rownames = FALSE,
      options = list(pageLength = 10, scrollX = TRUE)
    )
  })
  
  # ------------------------------------------------------------
  # EDA
  # ------------------------------------------------------------
  
  output$missing_table <- renderDT({
    missing_df <- data.frame(
      variable = names(filtered_df()),
      missing_count = as.numeric(sapply(filtered_df(), function(x) sum(is.na(x)))),
      missing_pct = as.numeric(round(sapply(filtered_df(), function(x) mean(is.na(x)) * 100), 2)),
      row.names = NULL
    ) %>%
      arrange(desc(missing_pct))
    
    datatable(
      missing_df,
      rownames = FALSE,
      options = list(pageLength = 10, scrollX = TRUE)
    )
  })
  
  output$position_count_plot <- renderPlotly({
    make_count_plot(filtered_df(), "position_type", "Top Position Types", "Position Type")
  })
  
  output$area_count_plot <- renderPlotly({
    make_count_plot(filtered_df(), "area_of_work", "Top Areas of Work", "Area of Work")
  })
  
  output$region_count_plot <- renderPlotly({
    make_count_plot(filtered_df(), "primary_region", "Postings by Region", "Primary Region")
  })
  
  output$education_count_plot <- renderPlotly({
    make_count_plot(filtered_df(), "required_education", "Required Education", "Required Education")
  })
  
  output$salary_hist <- renderPlotly({
    make_hist_plot(filtered_df(), "mid_salary", "Mid Salary Distribution", "Mid Salary", bins = 30)
  })
  
  output$log_salary_hist <- renderPlotly({
    make_hist_plot(filtered_df(), "log_mid_salary", "Log Salary Distribution", "Log Mid Salary", bins = 30)
  })
  
  output$salary_range_hist <- renderPlotly({
    make_hist_plot(filtered_df(), "salary_range", "Salary Range Distribution", "Salary Range", bins = 30)
  })
  
  output$corr_salary_plot <- renderPlotly({
    data_now <- filtered_df()
    
    if (!("log_mid_salary" %in% names(data_now))) {
      return(empty_plot("log_mid_salary is not available."))
    }
    
    numeric_df <- data_now %>%
      select(where(is.numeric))
    
    if (!("log_mid_salary" %in% names(numeric_df))) {
      return(empty_plot("log_mid_salary is not numeric."))
    }
    
    if (ncol(numeric_df) < 2 || nrow(numeric_df) < 3) {
      return(empty_plot("Not enough numeric data for correlation."))
    }
    
    target <- numeric_df$log_mid_salary
    
    if (sum(!is.na(target)) < 3 || sd(target, na.rm = TRUE) == 0) {
      return(empty_plot("Not enough variation in log salary for the selected filters."))
    }
    
    exclude_cols <- c(
      "log_mid_salary",
      "mid_salary",
      "min_salary",
      "max_salary",
      "salary_range",
      "salary_num"
    )
    
    candidate_cols <- setdiff(names(numeric_df), exclude_cols)
    
    if (length(candidate_cols) == 0) {
      return(empty_plot("No numeric feature columns available for correlation."))
    }
    
    corr_vals <- sapply(candidate_cols, function(col) {
      safe_cor(numeric_df[[col]], target)
    })
    
    corr_df <- data.frame(
      variable = names(corr_vals),
      correlation = as.numeric(corr_vals),
      row.names = NULL
    ) %>%
      filter(!is.na(correlation)) %>%
      mutate(abs_correlation = abs(correlation)) %>%
      arrange(desc(abs_correlation)) %>%
      head(15) %>%
      arrange(correlation)
    
    if (nrow(corr_df) == 0) {
      return(empty_plot("No valid correlations available for the selected filters."))
    }
    
    corr_df$variable <- factor(corr_df$variable, levels = corr_df$variable)
    
    p <- ggplot(
      corr_df,
      aes(
        x = correlation,
        y = variable,
        text = paste0("Variable: ", variable, "<br>Correlation: ", round(correlation, 3))
      )
    ) +
      geom_col(fill = "steelblue") +
      labs(
        title = "Correlations with Log Salary",
        x = "Correlation",
        y = "Variable"
      ) +
      base_theme()
    
    plotly_clean(p)
  })
  
  # ------------------------------------------------------------
  # Salary Exploration
  # ------------------------------------------------------------
  
  output$salary_by_area <- renderPlotly({
    make_median_bar_plot(
      filtered_df(),
      "area_of_work",
      "mid_salary",
      "Median Salary by Area",
      "Area of Work",
      "Median Salary"
    )
  })
  
  output$salary_by_position <- renderPlotly({
    make_horizontal_box_plot(
      filtered_df(),
      "position_type",
      "mid_salary",
      "Salary by Position Type",
      "Position Type",
      "Mid Salary"
    )
  })
  
  output$salary_by_region <- renderPlotly({
    make_median_bar_plot(
      filtered_df(),
      "primary_region",
      "mid_salary",
      "Median Salary by Region",
      "Primary Region",
      "Median Salary"
    )
  })
  
  output$salary_by_required_edu <- renderPlotly({
    make_horizontal_box_plot(
      filtered_df(),
      "required_education",
      "mid_salary",
      "Salary by Required Education",
      "Required Education",
      "Mid Salary"
    )
  })
  
  output$salary_by_preferred_edu <- renderPlotly({
    make_horizontal_box_plot(
      filtered_df(),
      "preferred_education",
      "mid_salary",
      "Salary by Preferred Education",
      "Preferred Education",
      "Mid Salary"
    )
  })
  
  # ------------------------------------------------------------
  # Feature Engineering
  # ------------------------------------------------------------
  
  output$engineered_feature_table <- renderDT({
    candidate_features <- c(
      "log_mid_salary",
      "salary_range",
      "required_edu_level",
      "preferred_edu_level",
      "edu_gap_preferred_minus_required",
      "title_len",
      "title_word_count",
      "n_skills_mentioned",
      "exp_years_min",
      "exp_years_max",
      "exp_years_any",
      "days_since_posted",
      "posted_month",
      "posted_dow",
      "posted_is_weekend",
      "n_states_listed",
      "is_multi_state_posting",
      "role_data",
      "role_engineering",
      "role_consulting",
      "role_security"
    )
    
    available <- candidate_features[candidate_features %in% names(df)]
    
    feature_table <- data.frame(
      feature = available,
      available_in_dataset = "Yes",
      row.names = NULL
    )
    
    datatable(feature_table, rownames = FALSE, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$skills_hist <- renderPlotly({
    make_hist_plot(
      filtered_df(),
      "n_skills_mentioned",
      "Skills Mentioned Distribution",
      "Number of Skills Mentioned",
      bins = 20
    )
  })
  
  output$exp_min_hist <- renderPlotly({
    make_hist_plot(
      filtered_df(),
      "exp_years_min",
      "Minimum Experience",
      "Minimum Years",
      bins = 15
    )
  })
  
  output$exp_max_hist <- renderPlotly({
    make_hist_plot(
      filtered_df(),
      "exp_years_max",
      "Maximum Experience",
      "Maximum Years",
      bins = 15
    )
  })
  
  output$role_summary_table <- renderDT({
    role_cols <- c(
      "role_data",
      "role_engineering",
      "role_consulting",
      "role_security"
    )
    
    role_cols <- role_cols[role_cols %in% names(filtered_df())]
    
    if (length(role_cols) == 0) {
      return(datatable(data.frame(Message = "No role indicator columns available."), rownames = FALSE))
    }
    
    role_summary <- data.frame(
      role_feature = role_cols,
      count = sapply(role_cols, function(x) sum(filtered_df()[[x]] == 1, na.rm = TRUE)),
      percentage = round(sapply(role_cols, function(x) mean(filtered_df()[[x]] == 1, na.rm = TRUE) * 100), 2),
      row.names = NULL
    )
    
    datatable(role_summary, rownames = FALSE, options = list(pageLength = 10))
  })
  
  # ------------------------------------------------------------
  # Unsupervised Learning
  # ------------------------------------------------------------
  
  output$pca_cluster_plot <- renderPlotly({
    required_cols <- c("PC1", "PC2", "cluster")
    
    if (!all(required_cols %in% names(filtered_df()))) {
      return(empty_plot("PCA columns or cluster column are not available in this dataset."))
    }
    
    plot_df <- filtered_df() %>%
      filter(!is.na(PC1), !is.na(PC2), !is.na(cluster))
    
    if (nrow(plot_df) == 0) {
      return(empty_plot("No PCA data available for the selected filters."))
    }
    
    p <- ggplot(
      plot_df,
      aes(
        x = PC1,
        y = PC2,
        color = cluster,
        text = paste(
          "Job Title:", job_title,
          "<br>Area:", area_of_work,
          "<br>Position:", position_type,
          "<br>Salary:", mid_salary
        )
      )
    ) +
      geom_point(alpha = 0.75, size = 2) +
      labs(
        title = "PCA by Cluster",
        x = "PC1",
        y = "PC2",
        color = "Cluster"
      ) +
      base_theme()
    
    plotly_clean(p)
  })
  
  output$salary_by_cluster <- renderPlotly({
    make_horizontal_box_plot(
      filtered_df(),
      "cluster",
      "mid_salary",
      "Salary by Cluster",
      "Cluster",
      "Mid Salary"
    )
  })
  
  output$cluster_summary_table <- renderDT({
    if (!has_cols(filtered_df(), c("cluster", "mid_salary"))) {
      return(datatable(data.frame(Message = "Cluster column is not available in this dataset."), rownames = FALSE))
    }
    
    cluster_summary <- filtered_df() %>%
      filter(!is.na(cluster), !is.na(mid_salary)) %>%
      group_by(cluster) %>%
      summarize(
        n_jobs = n(),
        mean_salary = round(mean(mid_salary, na.rm = TRUE), 0),
        median_salary = round(median(mid_salary, na.rm = TRUE), 0),
        avg_skills = if ("n_skills_mentioned" %in% names(filtered_df())) round(mean(n_skills_mentioned, na.rm = TRUE), 2) else NA,
        .groups = "drop"
      )
    
    datatable(cluster_summary, rownames = FALSE, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$outlier_table <- renderDT({
    if (!("is_outlier" %in% names(filtered_df()))) {
      return(
        datatable(
          data.frame(Message = "Outlier column is not available in this dataset."),
          rownames = FALSE,
          options = list(pageLength = 5)
        )
      )
    }
    
    outliers <- filtered_df() %>%
      filter(is_outlier == TRUE) %>%
      select(any_of(c(
        "job_title",
        "area_of_work",
        "position_type",
        "primary_region",
        "mid_salary",
        "cluster",
        "outlier_score"
      )))
    
    if (nrow(outliers) == 0) {
      return(datatable(data.frame(Message = "No outliers available for the selected filters."), rownames = FALSE))
    }
    
    datatable(outliers, rownames = FALSE, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # ------------------------------------------------------------
  # Modeling
  # ------------------------------------------------------------
  
  model_log_results <- data.frame(
    Model = c("Random Forest", "Gradient Boosting", "Ridge Regression"),
    Baseline_RMSE_Log = c(0.1934, 0.1812, 0.1916),
    Baseline_MAE_Log = c(0.1464, 0.1392, 0.1454),
    Baseline_R2 = c(0.7196, 0.7545, 0.7249),
    Tuned_CV_RMSE_Log = c(0.1922, 0.1816, 0.1865),
    Tuned_Test_RMSE_Log = c(0.1466, 0.1496, 0.1753),
    Tuned_Test_MAE_Log = c(0.1098, 0.1077, 0.1286),
    Tuned_Test_R2 = c(0.8568, 0.8508, 0.7953),
    Decision = c("Selected final model", "Strong alternative", "Linear baseline")
  )
  
  model_usd_results <- data.frame(
    Model = c("Random Forest", "Gradient Boosting", "Ridge Regression"),
    Tuned_Test_RMSE_USD = c(21727, 22270, 25541),
    Tuned_Test_MAE_USD = c(15310, 14929, 17818),
    Interpretation = c(
      "Lowest RMSE and selected final model",
      "Lowest MAE but slightly higher RMSE",
      "Highest error; used as baseline"
    )
  )
  
  output$model_log_table <- renderDT({
    datatable(
      model_log_results,
      rownames = FALSE,
      options = list(
        paging = FALSE,
        searching = FALSE,
        info = FALSE,
        scrollX = TRUE
      )
    )
  })
  
  output$model_usd_table <- renderDT({
    datatable(
      model_usd_results,
      rownames = FALSE,
      options = list(
        paging = FALSE,
        searching = FALSE,
        info = FALSE,
        scrollX = TRUE
      )
    )
  })
  
  output$rmse_plot <- renderPlotly({
    p <- ggplot(
      model_usd_results,
      aes(
        x = reorder(Model, Tuned_Test_RMSE_USD),
        y = Tuned_Test_RMSE_USD,
        fill = Model,
        text = paste0("Model: ", Model, "<br>RMSE: $", format(Tuned_Test_RMSE_USD, big.mark = ","))
      )
    ) +
      geom_col() +
      labs(
        title = "Tuned Test RMSE",
        x = "Model",
        y = "RMSE in USD"
      ) +
      base_theme() +
      theme(legend.position = "none")
    
    plotly_clean(p)
  })
  
  output$mae_plot <- renderPlotly({
    p <- ggplot(
      model_usd_results,
      aes(
        x = reorder(Model, Tuned_Test_MAE_USD),
        y = Tuned_Test_MAE_USD,
        fill = Model,
        text = paste0("Model: ", Model, "<br>MAE: $", format(Tuned_Test_MAE_USD, big.mark = ","))
      )
    ) +
      geom_col() +
      labs(
        title = "Tuned Test MAE",
        x = "Model",
        y = "MAE in USD"
      ) +
      base_theme() +
      theme(legend.position = "none")
    
    plotly_clean(p)
  })
  
  output$r2_plot <- renderPlotly({
    p <- ggplot(
      model_log_results,
      aes(
        x = reorder(Model, Tuned_Test_R2),
        y = Tuned_Test_R2,
        fill = Model,
        text = paste0("Model: ", Model, "<br>Tuned Test R-squared: ", Tuned_Test_R2)
      )
    ) +
      geom_col() +
      labs(
        title = "Tuned Test R-squared",
        x = "Model",
        y = "R-squared"
      ) +
      base_theme() +
      theme(legend.position = "none")
    
    plotly_clean(p)
  })
  
  output$final_model_text <- renderUI({
    HTML("
      <div class='model-note'>
        <b>Final Model Selection: Random Forest</b><br><br>

        Based on the unified model evaluation, <b>Random Forest</b> is selected as the final model.
        Random Forest achieved the lowest tuned test RMSE in dollar value, with an RMSE of 
        <b>$21,727</b>, compared with <b>$22,270</b> for Gradient Boosting and <b>$25,541</b> for 
        Ridge Regression.<br><br>

        Gradient Boosting achieved a slightly lower tuned test MAE, but its RMSE was slightly higher
        than Random Forest. Since RMSE penalizes larger prediction errors more heavily, Random Forest
        is preferred for reducing large salary prediction mistakes.<br><br>

        Ridge Regression performed worse than the two tree-based models because it is a linear model
        and cannot capture nonlinear relationships among job features as effectively.<br><br>

        Overall, Random Forest provides the best balance between predictive performance, robustness,
        and interpretability. It also provides feature importance, making it easier to explain which
        job characteristics drive salary predictions.<br><br>

        <b>Final choice:</b> Random Forest.
      </div>
    ")
  })
}

# ============================================================
# 5. Run App
# ============================================================

shinyApp(ui = ui, server = server)