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
  "randomForest" # Changed from ranger to match course materials 
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
final_data <- master_tbl |>
  mutate(across(-c(date, sp500_excess), lag, .names = "{.col}_lag1")) |> 
  select(date, sp500_excess, ends_with("_lag1")) |>
  slice(-1)

###########################################################
### --- 7. Define Predictor Sets for Analysis --- ###
###########################################################

# We define our 4 sets of predictors here

# Set 1: FF5 Only
ff5_predictors_lagged <- c("mkt_excess_lag1", "smb_lag1", "hml_lag1", "rmw_lag1", "cma_lag1")

# Set 2: Themes Only
theme_names <- all_themes_wide %>% select(-date) %>% colnames()
theme_predictors_lagged <- paste0(theme_names, "_lag1")

# Set 3: Factors Only
factor_names <- all_factors_wide %>% select(-date) %>% colnames()
factor_predictors_lagged <- paste0(factor_names, "_lag1")

# Set 4: All Predictors
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
# We learn parameters from the training set ONLY

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
  
  # Data for lm() and randomForest()
  train_data_subset <- train_data_proc %>% 
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
  
  # --- Finer grid for alpha ---
  alphas_to_try <- seq(0, 1, by = 0.1) 
  
  model_list <- list()
  validation_mse <- c()
  
  cat("Tuning Elastic Net (alpha grid):", alphas_to_try, "\n")
  
  for (a in alphas_to_try) {
    # Fit cv.glmnet for each alpha
    cv_fit <- cv.glmnet(x_train_subset, y_train, alpha = a, family = "gaussian")
    
    # Get validation MSE for this alpha
    preds_val <- predict(cv_fit, newx = x_val_subset, s = "lambda.min")
    mse_val <- mean((y_val - preds_val)^2)
    
    # Store the model and its validation MSE, named by alpha
    model_list[[as.character(a)]] <- cv_fit
    validation_mse[as.character(a)] <- mse_val
  }
  
  # --- Identify the specific models based on alpha ---
  
  # Model 2a: Ridge (alpha = 0)
  ridge_model <- model_list[["0"]]
  
  # Model 2b: Lasso (alpha = 1)
  lasso_model <- model_list[["1"]]
  
  # Model 2c: Best Tuned Elastic Net (alpha between 0 and 1)
  # We exclude 0 and 1 from the search
  enet_mses <- validation_mse[!names(validation_mse) %in% c("0", "1")]
  
  # Find the best alpha from this subset
  best_alpha_str <- names(which.min(enet_mses))
  best_alpha <- as.numeric(best_alpha_str)
  best_enet_model <- model_list[[best_alpha_str]]
  
  print(paste("Best Tuned Elastic Net Alpha (", analysis_label, "):", best_alpha))
  
  # Model 3: Random Forest
  set.seed(123) # for reproducibility
  rf_model <- randomForest(
    formula   = sp500_excess ~ ., 
    data      = train_data_subset,
    num.trees = 500,
    mtry      = floor(ncol(x_train_subset) / 3) 
  )
  
  # --- 3. Assess Models on TEST Data ---
  
  # OLS MSE
  preds_ols <- predict(ols_model, newdata = test_data_subset)
  mse_ols <- mean((y_test - preds_ols)^2)
  
  # Ridge MSE
  preds_ridge <- predict(ridge_model, newx = x_test_subset, s = "lambda.min")
  mse_ridge <- mean((y_test - preds_ridge)^2)
  
  # Lasso MSE
  preds_lasso <- predict(lasso_model, newx = x_test_subset, s = "lambda.min")
  mse_lasso <- mean((y_test - preds_lasso)^2)
  
  # Best Elastic Net MSE
  preds_enet <- predict(best_enet_model, newx = x_test_subset, s = "lambda.min")
  mse_enet <- mean((y_test - preds_enet)^2)
  
  # Random Forest MSE
  preds_rf <- predict(rf_model, newdata = test_data_subset)
  mse_rf <- mean((y_test - preds_rf)^2)
  
  # --- 4. Return Results ---
  results <- tibble(
    Analysis = analysis_label,
    Model = c(
      "OLS", 
      "Ridge (alpha=0)", 
      "Lasso (alpha=1)", 
      paste0("Elastic Net (a=", best_alpha, ")"), # Add best alpha to name
      "Random Forest"
    ),
    Test_MSE = c(mse_ols, mse_ridge, mse_lasso, mse_enet, mse_rf)
  )
  
  return(results)
}


#################################################
### --- 11. Run All Analyses & Compare --- ###
#################################################

# We now just call our function 4 times.

# Analysis 1: FF5 Only
results_ff5 <- run_analysis_pipeline(
  predictor_names = ff5_predictors_lagged,
  analysis_label = "FF5 Only"
)

# Analysis 2: Themes Only
results_themes <- run_analysis_pipeline(
  predictor_names = theme_predictors_lagged,
  analysis_label = "Themes Only"
)

# Analysis 3: Factors Only
results_factors <- run_analysis_pipeline(
  predictor_names = factor_predictors_lagged,
  analysis_label = "Factors Only"
)

# Analysis 4: All Predictors
results_all <- run_analysis_pipeline(
  predictor_names = all_predictors_lagged,
  analysis_label = "All Predictors"
)

# --- 12. Consolidate and Print Final Results ---

final_results_table <- bind_rows(
  results_ff5,
  results_themes,
  results_factors,
  results_all
)

print(final_results_table, n = Inf)