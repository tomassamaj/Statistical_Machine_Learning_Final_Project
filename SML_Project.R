# --- 1. Load All Necessary Packages ---

all_packages <- c(
  "dplyr", 
  "frenchdata",
  "ggplot2",
  "lubridate",
  "tidyverse",
  "quantmod",
  "caret",
  "glmnet",
  "randomForest",
  "gbm"  # <-- ADDED: For Gradient Boosting
)

options(repos = "https://cloud.r-project.org")

installed <- rownames(installed.packages())
for(pkg in all_packages) {
  if(! pkg %in% installed) install.packages(pkg)
}

invisible(lapply(all_packages, library, character.only = TRUE))

# --- 2. Set Random Seed and Working Directory ---
set.seed(123) # Set seed for reproducibility 
print(getwd())

#########################################
### Load and Prepare Fama-French Data ###
#########################################

start_date <- ymd("1993-01-01") 
end_date <- ymd("2024-12-31")

factors_ff5_monthly_raw <- download_french_data("Fama/French 5 Factors (2x3)")

factors_ff5_monthly <- factors_ff5_monthly_raw$subsets$data[[1]] |>
  mutate(
    date = rollforward(ymd(paste0(date, "01"))), 
    across(c(RF, `Mkt-RF`, SMB, HML, RMW, CMA), ~as.numeric(.) / 100),
    .keep = "none"
  ) |>
  rename_with(str_to_lower) |>
  rename(mkt_excess = `mkt-rf`) |>
  filter(date >= start_date & date <= end_date) |>
  slice(-1) # Drop the first row

########################################################################
### Download monthly SP500 returns from yahoo finance using quantmod ###
########################################################################

sp500_raw <- getSymbols("SPY", 
                        src = "yahoo", 
                        from = start_date, 
                        to = end_date, 
                        auto.assign = FALSE)

sp500_monthly_returns <- monthlyReturn(Ad(sp500_raw))

sp500_returns_tbl <- sp500_monthly_returns |>
  as_tibble(rownames = "date") |>
  mutate(date = ymd(date)) |>
  rename(sp500_return = monthly.returns) |>
  slice(-1) # Drop the first row

##########################################################################
### Load monthly factor and theme data from the Global factor database ###
##########################################################################

all_factors_monthly_vw_cap  <- read.csv("[usa]_[all_factors]_[monthly]_[vw_cap].csv")
all_themes_monthly_vw_cap   <- read.csv("[usa]_[all_themes]_[monthly]_[vw_cap].csv")

# --- Process Factors ---
all_factors_wide <- all_factors_monthly_vw_cap  %>%
  select(date, name, ret) %>%
  pivot_wider(names_from = name, values_from = ret) %>%
  mutate(date = as.Date(as.character(date))) %>%
  filter(date >= start_date & date <= end_date) %>%
  slice(-1) 

# --- Process Themes ---
all_themes_wide <- all_themes_monthly_vw_cap  %>%
  select(date, name, ret) %>%
  pivot_wider(names_from = name, values_from = ret) %>%
  mutate(date = as.Date(as.character(date))) %>%
  filter(date >= start_date & date <= end_date) %>%
  slice(-1)

###########################################
### Merge All Data into a Master Tibble ###
###########################################

master_tbl <- sp500_returns_tbl |>
  inner_join(factors_ff5_monthly, by = "date") |>
  inner_join(all_factors_wide, by = "date") |>
  inner_join(all_themes_wide, by = "date") |>
  mutate(sp500_excess = sp500_return - rf) |>
  select(date, sp500_excess, mkt_excess, smb, hml, rmw, cma, rf, everything(), -sp500_return)

# --- Create Predictive Lagged Dataset ---
# This version does NOT include interaction terms
final_data <- master_tbl |>
  mutate(across(-c(date, sp500_excess), lag, .names = "{.col}_lag1")) |> 
  select(date, sp500_excess, ends_with("_lag1")) |>
  slice(-1) |>
  drop_na() # Clean up any remaining NAs

###########################################################
### --- 7. Define Predictor Sets for Analysis --- ###
###########################################################

# Set 1: FF5 Only
ff5_predictors_lagged <- c("mkt_excess_lag1", "smb_lag1", "hml_lag1", "rmw_lag1", "cma_lag1")

# Set 2: Themes Only
theme_names <- all_themes_wide %>% select(-date) %>% colnames()
theme_predictors_lagged <- paste0(theme_names, "_lag1")

# Set 3: Factors Only
factor_names <- all_factors_wide %>% select(-date) %>% colnames()
factor_predictors_lagged <- paste0(factor_names, "_lag1")

# Set 4: All Predictors
# This definition now correctly grabs all predictors *without* any interactions
all_predictors_lagged <- final_data %>% 
  select(-date, -sp500_excess) %>% 
  colnames()

#################################################
### --- 8. Create Time-Series Splits (Train/Val/Test) --- ###
#################################################

n <- nrow(final_data)
train_end <- floor(0.50 * n)
val_end <- floor(0.75 * n)

train_data <- final_data |> slice(1:train_end)
val_data   <- final_data |> slice((train_end + 1):val_end)
test_data  <- final_data |> slice((val_end + 1):n)

# --- 9. Pre-processing: Standardize the Data ---
preproc_params <- preProcess(train_data[, -c(1, 2)], 
                             method = c("center", "scale"))

train_data_proc <- predict(preproc_params, train_data)
val_data_proc   <- predict(preproc_params, val_data)
test_data_proc  <- predict(preproc_params, test_data)

# Create full X/Y matrices
x_train_full <- as.matrix(train_data_proc[, -c(1, 2)])
y_train      <- train_data_proc$sp500_excess

x_val_full <- as.matrix(val_data_proc[, -c(1, 2)])
y_val      <- val_data_proc$sp500_excess

x_test_full <- as.matrix(test_data_proc[, -c(1, 2)])
y_test      <- test_data_proc$sp500_excess

#################################################
### --- 10. Define Reusable Analysis Function --- ###
#################################################

run_analysis_pipeline <- function(predictor_names, analysis_label) {
  
  cat("\n--- Running Analysis for:", analysis_label, "---\n")
  
  # --- 1. Create Data Subsets ---
  
  # Data for lm(), randomForest(), gbm()
  train_data_subset <- train_data_proc %>% 
    select(sp500_excess, all_of(predictor_names))
  val_data_subset <- val_data_proc %>% 
    select(sp500_excess, all_of(predictor_names))
  test_data_subset <- test_data_proc %>% 
    select(sp500_excess, all_of(predictor_names))
    
  # Matrices for glmnet()
  x_train_subset <- x_train_full[, predictor_names, drop = FALSE]
  x_val_subset   <- x_val_full[, predictor_names, drop = FALSE]
  x_test_subset  <- x_test_full[, predictor_names, drop = FALSE]

  # --- 2. Fit Models ---
  
  # Model 1: OLS (Baseline)
  ols_model <- lm(sp500_excess ~ ., data = train_data_subset)
  
  # Model 2: Tune Elastic Net (Ridge, Lasso, and ENET)
  alphas_to_try <- seq(0, 1, by = 0.1) 
  model_list <- list()
  validation_mse <- c()
  
  cat("Tuning Elastic Net (alpha grid):", alphas_to_try, "\n")
  
  for (a in alphas_to_try) {
    cv_fit <- cv.glmnet(x_train_subset, y_train, alpha = a, family = "gaussian")
    preds_val <- predict(cv_fit, newx = x_val_subset, s = "lambda.min")
    mse_val <- mean((y_val - preds_val)^2)
    model_list[[as.character(a)]] <- cv_fit
    validation_mse[as.character(a)] <- mse_val
  }
  
  ridge_model <- model_list[["0"]]
  lasso_model <- model_list[["1"]]
  
  enet_mses <- validation_mse[!names(validation_mse) %in% c("0", "1")]
  best_alpha_str <- names(which.min(enet_mses))
  best_alpha <- as.numeric(best_alpha_str)
  best_enet_model <- model_list[[best_alpha_str]]
  
  print(paste("Best Tuned Elastic Net Alpha (", analysis_label, "):", best_alpha))
  
  # Model 3: Random Forest
  set.seed(123)
  rf_model <- randomForest(
    formula   = sp500_excess ~ ., 
    data      = train_data_subset,
    num.trees = 500,
    mtry      = floor(ncol(x_train_subset) / 3) 
  )
  
  # Model 4: Gradient Boosting (GBM)
  cat("Tuning Gradient Boosting...\n")
  param_grid_gbm <- expand.grid(
    shrinkage = c(0.01, 0.1),
    interaction.depth = c(1, 2, 3),
    n.trees = c(100, 200, 300)
  )
  best_gbm_mse <- Inf
  best_gbm_params <- NULL
  
  for (i in 1:nrow(param_grid_gbm)) {
    params <- param_grid_gbm[i, ]
    set.seed(123)
    gbm_fit <- gbm(
      sp500_excess ~ ., data = train_data_subset, distribution = "gaussian",
      n.trees = params$n.trees, interaction.depth = params$interaction.depth,
      shrinkage = params$shrinkage, n.minobsinnode = 10, verbose = FALSE
    )
    preds_val_gbm <- predict(gbm_fit, newdata = val_data_subset, n.trees = params$n.trees)
    mse_val_gbm <- mean((y_val - preds_val_gbm)^2)
    if (mse_val_gbm < best_gbm_mse) {
      best_gbm_mse <- mse_val_gbm
      best_gbm_params <- params
    }
  }
  print(paste("Best GBM interaction.depth:", best_gbm_params$interaction.depth))
  print(paste("Best GBM shrinkage:", best_gbm_params$shrinkage))
  print(paste("Best GBM n.trees:", best_gbm_params$n.trees))
  
  set.seed(123)
  final_gbm_model <- gbm(
    sp500_excess ~ ., data = train_data_subset, distribution = "gaussian",
    n.trees = best_gbm_params$n.trees, interaction.depth = best_gbm_params$interaction.depth,
    shrinkage = best_gbm_params$shrinkage, n.minobsinnode = 10, verbose = FALSE
  )
  
  # --- 3. Assess Models on TEST Data ---
  
  # Get all predictions
  preds_ols <- predict(ols_model, newdata = test_data_subset)
  preds_ridge <- predict(ridge_model, newx = x_test_subset, s = "lambda.min")
  preds_lasso <- predict(lasso_model, newx = x_test_subset, s = "lambda.min")
  preds_enet <- predict(best_enet_model, newx = x_test_subset, s = "lambda.min")
  preds_rf <- predict(rf_model, newdata = test_data_subset)
  preds_gbm <- predict(final_gbm_model, newdata = test_data_subset, n.trees = best_gbm_params$n.trees)
  
  # Calculate MSEs
  mse_ols <- mean((y_test - preds_ols)^2)
  mse_ridge <- mean((y_test - as.vector(preds_ridge))^2)
  mse_lasso <- mean((y_test - as.vector(preds_lasso))^2)
  mse_enet <- mean((y_test - as.vector(preds_enet))^2)
  mse_rf <- mean((y_test - preds_rf)^2)
  mse_gbm <- mean((y_test - preds_gbm)^2)
  
  # --- 4. Return Results ---
  results <- tibble(
    Analysis = analysis_label,
    Model = c(
      "OLS", 
      "Ridge (alpha=0)", 
      "Lasso (alpha=1)", 
      paste0("Elastic Net (a=", best_alpha, ")"),
      "Random Forest",
      "Gradient Boosting (Tuned)"
    ),
    Test_MSE = c(mse_ols, mse_ridge, mse_lasso, mse_enet, mse_rf, mse_gbm)
  )
  
  # <-- MODIFIED: Create a tibble of predictions -->
  predictions_tbl <- tibble(
    date = test_data_proc$date, # Get dates from test set
    Actual = y_test,
    OLS = as.vector(preds_ols),
    Ridge = as.vector(preds_ridge),
    Lasso = as.vector(preds_lasso),
    Elastic_Net = as.vector(preds_enet),
    Random_Forest = as.vector(preds_rf),
    Gradient_Boosting = as.vector(preds_gbm)
  )
  
  # <-- MODIFIED: Return both results and predictions -->
  return(list(results = results, predictions = predictions_tbl))
}


#################################################
### --- 11. Run All Analyses & Compare --- ###
#################################################

# <-- MODIFIED: Store the full list output from each run -->

# Analysis 1: FF5 Only
analysis_output_ff5 <- run_analysis_pipeline(
  predictor_names = ff5_predictors_lagged,
  analysis_label = "FF5 Only"
)

# Analysis 2: Themes Only
analysis_output_themes <- run_analysis_pipeline(
  predictor_names = theme_predictors_lagged,
  analysis_label = "Themes Only"
)

# Analysis 3: Factors Only
analysis_output_factors <- run_analysis_pipeline(
  predictor_names = factor_predictors_lagged,
  analysis_label = "Factors Only"
)

# Analysis 4: All Predictors
analysis_output_all <- run_analysis_pipeline(
  predictor_names = all_predictors_lagged,
  analysis_label = "All Predictors"
)

# --- 12. Consolidate and Print Final Results ---

# <-- MODIFIED: Extract the $results component from each output -->
final_results_table <- bind_rows(
  analysis_output_ff5$results,
  analysis_output_themes$results,
  analysis_output_factors$results,
  analysis_output_all$results
)

print(final_results_table, n = Inf)


#################################################
### --- 13. Plot Test Set Predictions --- ###
#################################################

# --- Define a reusable plotting function ---
# This avoids repeating the same ggplot code four times

plot_cumulative_returns <- function(predictions_tbl, analysis_title) {
  
  # --- 1. Calculate Cumulative Returns ---
  cumulative_returns_tbl <- predictions_tbl %>%
    # Convert all excess returns to gross returns (1 + r)
    mutate(across(-date, ~ 1 + .)) %>%
    # Calculate the cumulative product for all columns
    mutate(across(-date, cumprod))

  # --- 2. Convert to Long Format for ggplot ---
  cumulative_returns_long <- cumulative_returns_tbl %>%
    pivot_longer(
      cols = -date,
      names_to = "Model",
      values_to = "Cumulative_Return"
    )

  # --- 3. Create the Plot ---
  # We use print() to ensure the plot displays when called from the script
  print(
    ggplot(cumulative_returns_long, aes(x = date, y = Cumulative_Return, color = Model)) +
      
      # Plot all the model prediction lines
      geom_line(data = . %>% filter(Model != "Actual"), alpha = 0.8) +
      
      # Add the "Actual" line on top, in black and slightly thicker
      geom_line(
        data = . %>% filter(Model == "Actual"), 
        color = "black", 
        linewidth = 0.75
      ) +
      
      labs(
        title = "Cumulative Returns of Model Predictions vs. Actual (Test Set)",
        subtitle = paste("Analysis:", analysis_title), # Use the function argument
        x = "Date",
        y = "Cumulative Return (1 + r)",
        color = "Legend"
      ) +
      
      # Add a horizontal line at 1.0 (breakeven)
      geom_hline(yintercept = 1.0, linetype = "dashed", color = "grey50") +
      
      # Use a log scale for the y-axis
      scale_y_log10() + 
      
      scale_color_manual(
        values = c(
          "Actual" = "black", 
          "OLS" = "deepskyblue3", 
          "Ridge" = "darkorange", 
          "Lasso" = "firebrick",
          "Elastic_Net" = "red", 
          "Random_Forest" = "forestgreen", 
          "Gradient_Boosting" = "darkviolet"
        )
      ) +
      theme_minimal() +
      theme(legend.position = "bottom")
  )
}

# --- Now, call the function for each of your four analyses ---

# Plot 1: FF5 Only
plot_cumulative_returns(
  predictions_tbl = analysis_output_ff5$predictions,
  analysis_title = "FF5 Predictors Only"
)

# Plot 2: Themes Only
plot_cumulative_returns(
  predictions_tbl = analysis_output_themes$predictions,
  analysis_title = "Theme Predictors Only"
)

# Plot 3: Factors Only
plot_cumulative_returns(
  predictions_tbl = analysis_output_factors$predictions,
  analysis_title = "Factor Predictors Only"
)

# Plot 4: All Predictors
plot_cumulative_returns(
  predictions_tbl = analysis_output_all$predictions,
  analysis_title = "All Predictors"
)
