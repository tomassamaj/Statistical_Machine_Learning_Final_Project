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
  "gbm",
  "nnet",
  "tibble",
  "corrplot"
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

#####################################################
### --- 7. Define Predictor Sets for Analysis --- ###
#####################################################

# Set 1: FF5 Only
ff5_predictors_lagged <- c("mkt_excess_lag1", "smb_lag1", "hml_lag1", "rmw_lag1", "cma_lag1")

# Set 1b: FF3 Only (a subset of FF5)
ff3_predictors_lagged <- c("mkt_excess_lag1", "smb_lag1", "hml_lag1")

# Set 2: Themes Only
theme_names <- all_themes_wide %>% select(-date) %>% colnames()
theme_predictors_lagged <- paste0(theme_names, "_lag1")

# Set 3: Factors Only
factor_names <- all_factors_wide %>% select(-date) %>% colnames()
factor_predictors_lagged <- paste0(factor_names, "_lag1")

# <-- MODIFIED: This is your new "All Predictors" set -->
# Set 4: All Predictors (FF5 + Themes + Factors)
# We use unique() to avoid duplicating any columns that might be in multiple sets
all_predictors_lagged <- unique(c(ff5_predictors_lagged, theme_predictors_lagged, factor_predictors_lagged))


#############################################################
### --- 8. Create Time-Series Splits (Train/Val/Test) --- ###
#############################################################

n <- nrow(final_data)
train_end <- floor(0.50 * n)
val_end <- floor(0.75 * n)

train_data <- final_data |> slice(1:train_end)
val_data   <- final_data |> slice((train_end + 1):val_end)
test_data  <- final_data |> slice((val_end + 1):n)

# --- 9. Pre-processing: Standardize the Data ---
# nnet is also sensitive to scaling, so we use the standardized data
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

#####################################################
### --- 10. Define Reusable Analysis Function --- ###
#####################################################

run_analysis_pipeline <- function(predictor_names, analysis_label) {
  
  cat("\n--- Running Analysis for:", analysis_label, "---\n")
  
  # --- 1. Create Data Subsets ---
  
  # Data for lm(), randomForest(), gbm(), nnet()
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
  
  # Model 1: Tune Elastic Net (Ridge, Lasso, and ENET)
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
  
  # <-- ADDED: Create a tibble of ENET tuning results -->
  enet_tuning_results <- tibble(
    alpha = alphas_to_try,
    Validation_MSE = validation_mse
  )
  
  # Model 2: Tune Random Forest
  cat("Tuning Random Forest (mtry)...\n")
  
  p_subset <- ncol(x_train_subset)
  # Create a reasonable grid of mtry values to test
  mtry_grid <- unique(floor(pmax(1, c(sqrt(p_subset), p_subset/3, p_subset/2))))
  
  best_rf_mse <- Inf
  best_mtry <- NA
  
  rf_tuning_results <- tibble(mtry = numeric(), Validation_MSE = numeric())
  
  for (m in mtry_grid) {
    set.seed(123)
    rf_fit <- randomForest(
      formula   = sp500_excess ~ ., 
      data      = train_data_subset,
      num.trees = 500,
      mtry      = m 
    )
    
    # Predict on VALIDATION data
    preds_val_rf <- predict(rf_fit, newdata = val_data_subset)
    mse_val_rf <- mean((y_val - preds_val_rf)^2)
    
    rf_tuning_results <- bind_rows(rf_tuning_results, tibble(mtry = m, Validation_MSE = mse_val_rf))
    
    if (mse_val_rf < best_rf_mse) {
      best_rf_mse <- mse_val_rf
      best_mtry <- m
    }
  }
  
  print(paste("Best RF mtry:", best_mtry))
  
  # Fit final RF on training data with best mtry
  set.seed(123)
  final_rf_model <- randomForest(
    formula   = sp500_excess ~ ., 
    data      = train_data_subset,
    num.trees = 500,
    mtry      = best_mtry,
    importance = TRUE # Need this for variable importance
  )
  
  # Model 3: Gradient Boosting (GBM)
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
  
  # Model 4: Tune Neural Network (size and decay)
  cat("Tuning Neural Network (nnet)...\n")
  
  # Hyperparameter grid for nnet
  size_grid <- c(3, 5, 10) # Number of hidden units
  lambdas_to_try <- seq(0, 0.1, by = 0.01) # decay parameter
  
  best_nn_mse <- Inf
  best_lambda_nn <- NA
  best_size_nn <- NA
  
  # Nested loop for size and decay
  for (s in size_grid) {
    for (lam in lambdas_to_try) {
      set.seed(123)
      nn_fit <- nnet::nnet(
        sp500_excess ~ .,
        data = train_data_subset,
        size = s,
        linout = TRUE, 
        decay = lam,   
        maxit = 1000,
        trace = FALSE,
        MaxNWts = 5000  # Increase weight limit
      )
      
      # Predict on VALIDATION data
      preds_val_nn <- predict(nn_fit, newdata = val_data_subset)
      mse_val_nn <- mean((y_val - preds_val_nn)^2)
      
      if (mse_val_nn < best_nn_mse) {
        best_nn_mse <- mse_val_nn
        best_lambda_nn <- lam
        best_size_nn <- s
      }
    }
  }
  
  print(paste("Best NN size:", best_size_nn))
  print(paste("Best NN decay (lambda):", best_lambda_nn))
  
  # Fit final NN on training data with best params
  set.seed(123)
  final_nn_model <- nnet::nnet(
    sp500_excess ~ .,
    data = train_data_subset,
    size = best_size_nn,
    linout = TRUE,
    decay = best_lambda_nn,
    maxit = 1000,
    trace = FALSE,
    MaxNWts = 5000  # Increase weight limit
  )
  
  # --- 3. Assess Models on TEST Data ---
  
  # Get all predictions
  preds_ridge <- predict(ridge_model, newx = x_test_subset, s = "lambda.min")
  preds_lasso <- predict(lasso_model, newx = x_test_subset, s = "lambda.min")
  preds_enet <- predict(best_enet_model, newx = x_test_subset, s = "lambda.min")
  preds_rf <- predict(final_rf_model, newdata = test_data_subset) 
  preds_gbm <- predict(final_gbm_model, newdata = test_data_subset, n.trees = best_gbm_params$n.trees)
  preds_nn <- predict(final_nn_model, newdata = test_data_subset) 
  
  # Calculate RMSEs (in %)
  rmse_ridge <- 100 * sqrt(mean((y_test - as.vector(preds_ridge))^2))
  rmse_lasso <- 100 * sqrt(mean((y_test - as.vector(preds_lasso))^2))
  rmse_enet  <- 100 * sqrt(mean((y_test - as.vector(preds_enet))^2))
  rmse_rf    <- 100 * sqrt(mean((y_test - preds_rf)^2))
  rmse_gbm   <- 100 * sqrt(mean((y_test - preds_gbm)^2))
  rmse_nn    <- 100 * sqrt(mean((y_test - as.vector(preds_nn))^2))
  
  # Calculate MAEs (in %)
  mae_ridge <- 100 * mean(abs(y_test - as.vector(preds_ridge)))
  mae_lasso <- 100 * mean(abs(y_test - as.vector(preds_lasso)))
  mae_enet  <- 100 * mean(abs(y_test - as.vector(preds_enet)))
  mae_rf    <- 100 * mean(abs(y_test - preds_rf))
  mae_gbm   <- 100 * mean(abs(y_test - preds_gbm))
  mae_nn    <- 100 * mean(abs(y_test - as.vector(preds_nn)))
  
  # --- 4. Extract Variable Importance ---
  
  # Random Forest Importance
  imp_rf_df <- as.data.frame(randomForest::importance(final_rf_model))
  imp_rf_df <- tibble::rownames_to_column(imp_rf_df, "Variable") %>% 
    select(Variable, Importance = IncNodePurity)
  
  # GBM Importance
  imp_gbm_df <- as.data.frame(gbm::summary.gbm(final_gbm_model, plotit = FALSE))
  imp_gbm_df <- imp_gbm_df %>% select(Variable = var, Importance = rel.inf)
  
  # --- 5. Return Results ---
  results <- tibble(
    Analysis = analysis_label,
    Model = c(
      "Ridge (alpha=0)", 
      "Lasso (alpha=1)", 
      paste0("Elastic Net (a=", best_alpha, ")"),
      paste0("Random Forest (mtry=", best_mtry, ")"),
      "Gradient Boosting (Tuned)",
      paste0("NN (size=", best_size_nn, ", decay=", best_lambda_nn, ")")
    ),
    Test_RMSE_pct = c(rmse_ridge, rmse_lasso, rmse_enet, rmse_rf, rmse_gbm, rmse_nn),
    Test_MAE_pct = c(mae_ridge, mae_lasso, mae_enet, mae_rf, mae_gbm, mae_nn)
  )
  
  # Create a tibble of predictions
  predictions_tbl <- tibble(
    date = test_data_proc$date, # Get dates from test set
    Actual = y_test,
    Ridge = as.vector(preds_ridge),
    Lasso = as.vector(preds_lasso),
    Elastic_Net = as.vector(preds_enet),
    Random_Forest = as.vector(preds_rf),
    Gradient_Boosting = as.vector(preds_gbm),
    Neural_Network = as.vector(preds_nn) 
  )
  
  # <-- MODIFIED: Return all results, including tuning data -->
  return(list(
    results = results, 
    predictions = predictions_tbl,
    importance_rf = imp_rf_df,
    importance_gbm = imp_gbm_df,
    enet_tuning = enet_tuning_results,
    rf_tuning = rf_tuning_results # <-- ADDED
  ))
}


##############################################
### --- 11. Run All Analyses & Compare --- ###
##############################################

# Analysis 1: FF5 Only
analysis_output_ff5 <- run_analysis_pipeline(
  predictor_names = ff5_predictors_lagged,
  analysis_label = "FF5 Only"
)

# Analysis 1b: FF3 Only
analysis_output_ff3 <- run_analysis_pipeline(
  predictor_names = ff3_predictors_lagged,
  analysis_label = "FF3 Only"
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

# <-- MODIFIED: This now runs the "All Predictors" analysis -->
# Analysis 4: All Predictors
analysis_output_all <- run_analysis_pipeline(
  predictor_names = all_predictors_lagged,
  analysis_label = "All Predictors"
)

# --- 12. Consolidate and Print Final Results ---

final_results_table <- bind_rows(
  analysis_output_ff5$results,
  analysis_output_ff3$results, 
  analysis_output_themes$results,
  analysis_output_factors$results,
  analysis_output_all$results # <-- MODIFIED
)

# The final table will now include tuned model names, Test_RMSE_pct and Test_MAE_pct
print(final_results_table, n = Inf)


##############################################################
### --- 13. Plot Test Set Predicted vs. Actual Returns --- ###
##############################################################

# --- Define a reusable plotting function for raw returns ---
plot_predicted_returns <- function(predictions_tbl, analysis_title) {
  
  # --- 1. Convert to Long Format for ggplot ---
  predictions_long <- predictions_tbl %>%
    pivot_longer(
      cols = -date,
      names_to = "Model",
      values_to = "Excess_Return"
    )
  
  # --- 2. Create the Plot ---
  print(
    ggplot(predictions_long, aes(x = date, y = Excess_Return, color = Model)) +
      
      geom_line(data = . %>% filter(Model != "Actual"), alpha = 0.7, linewidth = 0.5) +
      
      geom_line(
        data = . %>% filter(Model == "Actual"), 
        color = "black", 
        linewidth = 0.75
      ) +
      
      labs(
        title = "Predicted vs. Actual Monthly Returns (Test Set)",
        subtitle = paste("Analysis:", analysis_title), 
        x = "Date",
        y = "S&P 500 Excess Return",
        color = "Legend"
      ) +
      
      geom_hline(yintercept = 0.0, linetype = "dashed", color = "grey50") +
      
      scale_color_manual(
        values = c(
          "Actual" = "black",
          "Ridge" = "darkorange", 
          "Lasso" = "firebrick",
          "Elastic_Net" = "red", 
          "Random_Forest" = "forestgreen", 
          "Gradient_Boosting" = "darkviolet",
          "Neural_Network" = "gold3"
        )
      ) +
      theme_minimal() +
      theme(legend.position = "bottom")
  )
}

# --- Call the function for each analysis ---

# Plot 1: FF5 Only
plot_predicted_returns(
  predictions_tbl = analysis_output_ff5$predictions,
  analysis_title = "FF5 Predictors Only"
)

# Plot 2: FF3 Only
plot_predicted_returns(
  predictions_tbl = analysis_output_ff3$predictions,
  analysis_title = "FF3 Predictors Only"
)

# Plot 3: Themes Only
plot_predicted_returns(
  predictions_tbl = analysis_output_themes$predictions,
  analysis_title = "Theme Predictors Only"
)

# Plot 4: Factors Only
plot_predicted_returns(
  predictions_tbl = analysis_output_factors$predictions,
  analysis_title = "Factor Predictors Only"
)

# <-- MODIFIED: This now plots the "All Predictors" results -->
# Plot 5: All Predictors
plot_predicted_returns(
  predictions_tbl = analysis_output_all$predictions,
  analysis_title = "All Predictors"
)


####################################################
### --- 14. Plot Test Set Cumulative Returns --- ###
####################################################

# --- Define a reusable plotting function ---
plot_cumulative_returns <- function(predictions_tbl, analysis_title) {
  
  # --- 1. Calculate Cumulative Returns ---
  cumulative_returns_tbl <- predictions_tbl %>%
    mutate(across(-date, ~ 1 + .)) %>%
    mutate(across(-date, cumprod))
  
  # --- 2. Convert to Long Format for ggplot ---
  cumulative_returns_long <- cumulative_returns_tbl %>%
    pivot_longer(
      cols = -date,
      names_to = "Model",
      values_to = "Cumulative_Return"
    )
  
  # --- 3. Create the Plot ---
  print(
    ggplot(cumulative_returns_long, aes(x = date, y = Cumulative_Return, color = Model)) +
      
      geom_line(data = . %>% filter(Model != "Actual"), alpha = 0.8) +
      
      geom_line(
        data = . %>% filter(Model == "Actual"), 
        color = "black", 
        linewidth = 0.75
      ) +
      
      labs(
        title = "Cumulative Returns of Model Predictions vs. Actual (Test Set)",
        subtitle = paste("Analysis:", analysis_title), 
        x = "Date",
        y = "Cumulative Return",
        color = "Legend"
      ) +
      
      geom_hline(yintercept = 1.0, linetype = "dashed", color = "grey50") +
      
      scale_y_log10() + 
      
      scale_color_manual(
        values = c(
          "Actual" = "black",
          "Ridge" = "darkorange", 
          "Lasso" = "firebrick",
          "Elastic_Net" = "red", 
          "Random_Forest" = "forestgreen", 
          "Gradient_Boosting" = "darkviolet",
          "Neural_Network" = "gold3" 
        )
      ) +
      theme_minimal() +
      theme(legend.position = "bottom")
  )
}

# --- Call the function for each of your analyses ---

# Plot 1: FF5 Only
plot_cumulative_returns(
  predictions_tbl = analysis_output_ff5$predictions,
  analysis_title = "FF5 Predictors Only"
)

# Plot 2: FF3 Only
plot_cumulative_returns(
  predictions_tbl = analysis_output_ff3$predictions,
  analysis_title = "FF3 Predictors Only"
)

# Plot 3: Themes Only
plot_cumulative_returns(
  predictions_tbl = analysis_output_themes$predictions,
  analysis_title = "Theme Predictors Only"
)

# Plot 4: Factors Only
plot_cumulative_returns(
  predictions_tbl = analysis_output_factors$predictions,
  analysis_title = "Factor Predictors Only"
)

# Plot 5: All Predictors
plot_cumulative_returns(
  predictions_tbl = analysis_output_all$predictions,
  analysis_title = "All Predictors"
)


############################################
### --- 15. Plot Variable Importance --- ###
############################################

# --- Define a reusable plotting function for variable importance ---
plot_variable_importance <- function(importance_df, model_title) {
  
  # Get top 5 variables
  top_10_vars <- importance_df %>%
    arrange(desc(Importance)) %>%
    slice_head(n = 5)
  
  # Create the plot
  print(
    ggplot(top_10_vars, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(
        title = paste("Most Important Predictors", model_title),
        x = "Predictor",
        y = "Importance (IncNodePurity / Rel. Inf.)"
      ) +
      theme_minimal()
  )
}

# Random Forest Importance (All Factors)
plot_variable_importance(
  importance_df = analysis_output_factors$importance_rf,
  model_title = "Random Forest (All Factors)"
)

# Gradient Boosting Importance (All Factors)
plot_variable_importance(
  importance_df = analysis_output_factors$importance_gbm,
  model_title = "Gradient Boosting (All Factors)"
)

# Random Forest Importance (Themes)
plot_variable_importance(
  importance_df = analysis_output_themes$importance_rf,
  model_title = "Random Forest (Themes)"
)

# Gradient Boosting Importance (Themes)
plot_variable_importance(
  importance_df = analysis_output_themes$importance_gbm,
  model_title = "Gradient Boosting (Themes)"
)

##############################################
### --- 16. Plot Hyperparameter Tuning --- ###
##############################################

# --- Define a reusable plotting function for ENET tuning ---
plot_enet_tuning <- function(tuning_tbl, analysis_title) {
  
  # Find the best alpha (excluding 0 and 1)
  best_enet_alpha <- tuning_tbl %>%
    filter(alpha > 0, alpha < 1) %>%
    filter(Validation_MSE == min(Validation_MSE)) %>%
    pull(alpha)
  
  # Get the MSE for Ridge and Lasso
  ridge_mse <- tuning_tbl %>% filter(alpha == 0) %>% pull(Validation_MSE)
  lasso_mse <- tuning_tbl %>% filter(alpha == 1) %>% pull(Validation_MSE)
  
  print(
    ggplot(tuning_tbl, aes(x = alpha, y = Validation_MSE)) +
      geom_line() +
      geom_point(size = 2) +
      
      # <-- MODIFIED: Use annotate() to fix warnings -->
      # Highlight Ridge
      annotate("point", x = 0, y = ridge_mse, color = "darkorange", size = 4) +
      annotate("text", x = 0.05, y = ridge_mse, label = "Ridge", hjust = 0, color = "darkorange") +
      
      # Highlight Lasso
      annotate("point", x = 1, y = lasso_mse, color = "firebrick", size = 4) +
      annotate("text", x = 0.95, y = lasso_mse, label = "Lasso", hjust = 1, color = "firebrick") +
      
      # Highlight Best Elastic Net
      geom_vline(xintercept = best_enet_alpha, linetype = "dashed", color = "red") +
      
      labs(
        title = "Elastic Net Tuning (Validation Set MSE vs. Alpha)",
        subtitle = paste("Analysis:", analysis_title),
        x = "Alpha (0 = Ridge, 1 = Lasso)",
        y = "Validation Set MSE"
      ) +
      theme_minimal()
  )
}

# --- Plot tuning for the "All Predictors" analysis ---
plot_enet_tuning(
  tuning_tbl = analysis_output_all$enet_tuning,
  analysis_title = "All Predictors"
)

# Plot tuning for the "FF5 Only" analysis
plot_enet_tuning(
  tuning_tbl = analysis_output_ff5$enet_tuning,
  analysis_title = "FF5 Predictors Only"
)


#################################################
### --- 17. Plot Lasso Variable Selection --- ###
#################################################

# We want to visualize how Lasso performs variable selection
# on the "Themes Only" dataset. We fit a new glmnet model
# on the *full training set* (train + validation) to get a stable path.


# 1. Combine Train + Validation data for a richer plot
# We use the full 75% of data designated for training/tuning
x_train_val_full <- rbind(x_train_full, x_val_full)
y_train_val      <- c(y_train, y_val)

# 2. Subset to just the "Themes" predictors
x_train_val_themes <- x_train_val_full[, theme_predictors_lagged, drop = FALSE]

# 3. Fit the Lasso (alpha=1) model
# We let glmnet create its full sequence of lambdas
lasso_path_model <- glmnet(
  x_train_val_themes,
  y_train_val,
  alpha = 1,
  family = "gaussian"
)

# 4. Create the coefficient path plot
# label = TRUE adds the variable names (theme names) to the plot
plot(lasso_path_model, xvar = "lambda", label = TRUE)
title(
  main = "Lasso Coefficient Path (Themes Predictors)", 
  line = 2.5
)

#################################################
### --- 17. Plot RF Hyperparameter Tuning --- ###
#################################################

# --- Define a reusable plotting function for RF mtry tuning ---
plot_rf_tuning <- function(tuning_tbl, analysis_title) {
  
  # Find the best mtry
  best_rf_mtry <- tuning_tbl %>%
    filter(Validation_MSE == min(Validation_MSE)) %>%
    pull(mtry)
  
  print(
    ggplot(tuning_tbl, aes(x = mtry, y = Validation_MSE)) +
      geom_line() +
      geom_point(size = 2) +
      
      # Highlight Best mtry
      geom_vline(xintercept = best_rf_mtry, linetype = "dashed", color = "forestgreen") +
      
      labs(
        title = "Random Forest Tuning (Validation Set MSE vs. mtry)",
        subtitle = paste("Analysis:", analysis_title),
        x = "mtry (Number of variables per split)",
        y = "Validation Set MSE"
      ) +
      theme_minimal()
  )
}

# --- Plot tuning for the "FF5 Only" analysis ---
plot_rf_tuning(
  tuning_tbl = analysis_output_ff5$rf_tuning,
  analysis_title = "FF5 Predictors Only"
)

# --- Plot tuning for the "All Factors" analysis ---
plot_rf_tuning(
  tuning_tbl = analysis_output_factors$rf_tuning,
  analysis_title = "All Factors"
)

# --- Plot tuning for the "All Themes" analysis ---
plot_rf_tuning(
  tuning_tbl = analysis_output_themes$rf_tuning,
  analysis_title = "All Themes"
)


#########################################
### --- 18. Plot Correlation Matrices --- ###
#########################################

# --- Define a reusable plotting function ---
plot_correlation_matrix <- function(data, title) {
  
  # 1. Compute Correlation Matrix
  # We remove the 'date' column and use only numeric columns
  cor_matrix <- cor(data %>% select(-date), use = "pairwise.complete.obs")
  
  # 2. Plot Upper Triangle
  # 'tl.cex' controls text size (smaller for large matrices)
  # 'tl.col' makes text black
  # 'type="upper"' shows only the upper triangle
  
  corrplot(cor_matrix, 
           method = "color", 
           type = "upper", 
           order = "hclust", # Group correlated variables together
           tl.col = "black", 
           tl.srt = 45, 
           title = title,
           mar = c(0,0,2,0), # Add margin for title
           tl.cex = 0.6      # Make labels smaller to fit
  )
}

# --- 1. FF5 Factors Correlation ---
# We use the original monthly data (before lagging)
plot_correlation_matrix(
  data = factors_ff5_monthly, 
  title = "Correlation: Fama-French 5 Factors"
)

# --- 2. All Themes Correlation ---
plot_correlation_matrix(
  data = all_themes_wide, 
  title = "Correlation: All 13 Themes"
)

# --- 3. All Factors Correlation ---
# This will be a very large plot (153x153)
# We might need to make the text extremely small or remove it
plot_correlation_matrix(
  data = all_factors_wide, 
  title = "Correlation: All 153 Factors"
)
