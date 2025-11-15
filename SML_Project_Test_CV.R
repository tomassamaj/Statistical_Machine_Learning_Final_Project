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
  "tibble" # For rownames_to_column
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

# Set 1b: FF3 Only (a subset of FF5)
ff3_predictors_lagged <- c("mkt_excess_lag1", "smb_lag1", "hml_lag1")

# Set 2: Themes Only
theme_names <- all_themes_wide %>% select(-date) %>% colnames()
theme_predictors_lagged <- paste0(theme_names, "_lag1")

# Set 3: Factors Only
factor_names <- all_factors_wide %>% select(-date) %>% colnames()
factor_predictors_lagged <- paste0(factor_names, "_lag1")

# Set 4: All Predictors (FF5 + Themes + Factors)
all_predictors_lagged <- unique(c(ff5_predictors_lagged, theme_predictors_lagged, factor_predictors_lagged))


#################################################
### --- 8. Create Time-Series Splits (Train/Test) --- ###
#################################################

# <-- MODIFIED: Switched to 70/30 Train/Test split -->
n <- nrow(final_data)
train_end <- floor(0.70 * n)

train_data <- final_data |> slice(1:train_end)
test_data  <- final_data |> slice((train_end + 1):n)

# --- 9. Pre-processing: Standardize the Data ---
preproc_params <- preProcess(train_data[, -c(1, 2)], 
                             method = c("center", "scale"))

train_data_proc <- predict(preproc_params, train_data)
test_data_proc  <- predict(preproc_params, test_data)

# Create full X/Y matrices and vectors for all models
x_train_full <- as.matrix(train_data_proc[, -c(1, 2)])
y_train      <- train_data_proc$sp500_excess

x_test_full <- as.matrix(test_data_proc[, -c(1, 2)])
y_test      <- test_data_proc$sp500_excess

# <-- ADDED: Define Time-Series CV Folds -->
# We create the folds on the *training set* for hyperparameter tuning
window_length <- 60  # 5 years
horizon       <- 24  # 2 years
set.seed(123)
cv_folds <- createTimeSlices(
  1:nrow(train_data_proc), # Use the 70% training set
  initialWindow = window_length,
  horizon       = horizon,
  fixedWindow   = TRUE # Rolling window
)


#################################################
### --- 10. Define Reusable Analysis Function --- ###
#################################################

# <-- MODIFIED: This function now uses TS-CV for tuning -->
run_analysis_pipeline <- function(predictor_names, analysis_label, cv_folds) {
  
  cat("\n--- Running Analysis for:", analysis_label, "---\n")
  
  # --- 1. Create Data Subsets ---
  
  # Data for lm(), randomForest(), gbm(), nnet()
  train_data_subset <- train_data_proc %>% 
    select(sp500_excess, all_of(predictor_names))
  test_data_subset <- test_data_proc %>% 
    select(sp500_excess, all_of(predictor_names))
  
  # Matrices for glmnet()
  x_train_subset <- x_train_full[, predictor_names, drop = FALSE]
  x_test_subset  <- x_test_full[, predictor_names, drop = FALSE]
  
  # --- 2. Fit Models ---
  
  # --- Model 1: Tune Elastic Net ---
  alphas_to_try <- seq(0, 1, by = 0.1) 
  best_mean_mse <- Inf
  best_alpha <- NA
  best_lambda <- NA
  
  cat("Tuning Elastic Net (alpha grid):", alphas_to_try, "\n")
  
  for (a in alphas_to_try) {
    fold_mses <- c()
    fold_lambdas <- c()
    
    for (i in seq_along(cv_folds$train)) {
      train_idx <- cv_folds$train[[i]]
      val_idx   <- cv_folds$test[[i]]
      
      cv_fit <- cv.glmnet(x_train_subset[train_idx, ], y_train[train_idx], 
                          alpha = a, family = "gaussian")
      
      fold_lambdas <- c(fold_lambdas, cv_fit$lambda.1se)
      
      preds_val <- predict(cv_fit, newx = x_train_subset[val_idx, ], s = "lambda.1se")
      fold_mses <- c(fold_mses, mean((y_train[val_idx] - preds_val)^2))
    }
    
    mean_fold_mse <- mean(fold_mses, na.rm = TRUE)
    
    if (mean_fold_mse < best_mean_mse) {
      best_mean_mse <- mean_fold_mse
      best_alpha <- a
      best_lambda <- median(fold_lambdas, na.rm = TRUE)
    }
  }
  
  print(paste("Best Tuned Elastic Net Alpha (", analysis_label, "):", best_alpha))
  print(paste("Best Tuned Elastic Net Lambda (", analysis_label, "):", best_lambda))
  
  # Fit final ENET models on *full training set*
  ridge_model <- glmnet(x_train_subset, y_train, alpha = 0, family = "gaussian", lambda = best_lambda)
  lasso_model <- glmnet(x_train_subset, y_train, alpha = 1, family = "gaussian", lambda = best_lambda)
  best_enet_model <- glmnet(x_train_subset, y_train, alpha = best_alpha, family = "gaussian", lambda = best_lambda)
  
  # --- Model 2: Tune Random Forest ---
  cat("Tuning Random Forest (mtry)...\n")
  
  p_subset <- ncol(x_train_subset)
  mtry_grid <- unique(floor(pmax(1, c(sqrt(p_subset), p_subset/3, p_subset/2))))
  
  best_mean_mse_rf <- Inf
  best_mtry <- NA
  
  for (m in mtry_grid) {
    fold_mses_rf <- c()
    for (i in seq_along(cv_folds$train)) {
      train_idx <- cv_folds$train[[i]]
      val_idx   <- cv_folds$test[[i]]
      
      set.seed(123)
      rf_fit <- randomForest(
        x = x_train_subset[train_idx, ],
        y = y_train[train_idx],
        num.trees = 500,
        mtry      = m 
      )
      
      preds_val_rf <- predict(rf_fit, newdata = x_train_subset[val_idx, ])
      fold_mses_rf <- c(fold_mses_rf, mean((y_train[val_idx] - preds_val_rf)^2))
    }
    
    mean_fold_mse_rf <- mean(fold_mses_rf, na.rm = TRUE)
    
    if (mean_fold_mse_rf < best_mean_mse_rf) {
      best_mean_mse_rf <- mean_fold_mse_rf
      best_mtry <- m
    }
  }
  
  print(paste("Best RF mtry:", best_mtry))
  
  # Fit final RF on training data with best mtry
  set.seed(123)
  final_rf_model <- randomForest(
    x = x_train_subset,
    y = y_train,
    num.trees = 500,
    mtry      = best_mtry,
    importance = TRUE 
  )
  
  # --- Model 3: Tune Gradient Boosting (GBM) ---
  cat("Tuning Gradient Boosting...\n")
  param_grid_gbm <- expand.grid(
    shrinkage = c(0.01, 0.1),
    interaction.depth = c(1, 2, 3),
    n.trees = c(100, 200, 300)
  )
  best_mean_mse_gbm <- Inf
  best_gbm_params <- NULL
  
  for (i in 1:nrow(param_grid_gbm)) {
    params <- param_grid_gbm[i, ]
    fold_mses_gbm <- c()
    
    for (j in seq_along(cv_folds$train)) {
      train_idx <- cv_folds$train[[j]]
      val_idx   <- cv_folds$test[[j]]
      
      set.seed(123)
      gbm_fit <- gbm(
        sp500_excess ~ ., 
        data = train_data_subset[train_idx, ], 
        distribution = "gaussian",
        n.trees = params$n.trees, 
        interaction.depth = params$interaction.depth,
        shrinkage = params$shrinkage, 
        n.minobsinnode = 10, 
        verbose = FALSE
      )
      preds_val_gbm <- predict(gbm_fit, newdata = train_data_subset[val_idx, ], 
                               n.trees = params$n.trees)
      fold_mses_gbm <- c(fold_mses_gbm, mean((y_train[val_idx] - preds_val_gbm)^2))
    }
    
    mean_fold_mse_gbm <- mean(fold_mses_gbm, na.rm = TRUE)
    
    if (mean_fold_mse_gbm < best_mean_mse_gbm) {
      best_mean_mse_gbm <- mean_fold_mse_gbm
      best_gbm_params <- params
    }
  }
  
  print(paste("Best GBM interaction.depth:", best_gbm_params$interaction.depth))
  print(paste("Best GBM shrinkage:", best_gbm_params$shrinkage))
  print(paste("Best GBM n.trees:", best_gbm_params$n.trees))
  
  # Fit final GBM on training data with best params
  set.seed(123)
  final_gbm_model <- gbm(
    sp500_excess ~ ., data = train_data_subset, distribution = "gaussian",
    n.trees = best_gbm_params$n.trees, 
    interaction.depth = best_gbm_params$interaction.depth,
    shrinkage = best_gbm_params$shrinkage, 
    n.minobsinnode = 10, 
    verbose = FALSE
  )
  
  # --- Model 4: Tune Neural Network (nnet) ---
  cat("Tuning Neural Network (nnet)...\n")
  
  size_grid <- c(3, 5, 10) 
  lambdas_to_try <- seq(0, 0.1, by = 0.01) 
  
  best_mean_mse_nn <- Inf
  best_lambda_nn <- NA
  best_size_nn <- NA
  
  for (s in size_grid) {
    for (lam in lambdas_to_try) {
      fold_mses_nn <- c()
      
      for (k in seq_along(cv_folds$train)) {
        train_idx <- cv_folds$train[[k]]
        val_idx   <- cv_folds$test[[k]]
        
        set.seed(123)
        nn_fit <- nnet::nnet(
          sp500_excess ~ .,
          data = train_data_subset[train_idx, ],
          size = s,
          linout = TRUE, 
          decay = lam,   
          maxit = 1000,
          trace = FALSE,
          MaxNWts = 5000 
        )
        
        preds_val_nn <- predict(nn_fit, newdata = train_data_subset[val_idx, ])
        fold_mses_nn <- c(fold_mses_nn, mean((y_train[val_idx] - preds_val_nn)^2))
      }
      
      mean_fold_mse_nn <- mean(fold_mses_nn, na.rm = TRUE)
      
      if (mean_fold_mse_nn < best_mean_mse_nn) {
        best_mean_mse_nn <- mean_fold_mse_nn
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
    MaxNWts = 5000
  )
  
  # --- 3. Assess Models on TEST Data ---
  
  # Get all predictions
  preds_ridge <- predict(ridge_model, newx = x_test_subset)
  preds_lasso <- predict(lasso_model, newx = x_test_subset)
  preds_enet <- predict(best_enet_model, newx = x_test_subset)
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
  
  # Return all results
  return(list(
    results = results, 
    predictions = predictions_tbl,
    importance_rf = imp_rf_df,
    importance_gbm = imp_gbm_df
  ))
}


#################################################
### --- 11. Run All Analyses & Compare --- ###
#################################################

# <-- MODIFIED: Pass the cv_folds to the function -->

# Analysis 1: FF5 Only
analysis_output_ff5 <- run_analysis_pipeline(
  predictor_names = ff5_predictors_lagged,
  analysis_label = "FF5 Only",
  cv_folds = cv_folds
)

# Analysis 1b: FF3 Only
analysis_output_ff3 <- run_analysis_pipeline(
  predictor_names = ff3_predictors_lagged,
  analysis_label = "FF3 Only",
  cv_folds = cv_folds
)

# Analysis 2: Themes Only
analysis_output_themes <- run_analysis_pipeline(
  predictor_names = theme_predictors_lagged,
  analysis_label = "Themes Only",
  cv_folds = cv_folds
)

# Analysis 3: Factors Only
analysis_output_factors <- run_analysis_pipeline(
  predictor_names = factor_predictors_lagged,
  analysis_label = "Factors Only",
  cv_folds = cv_folds
)

# Analysis 4: All Predictors
analysis_output_all <- run_analysis_pipeline(
  predictor_names = all_predictors_lagged,
  analysis_label = "All Predictors",
  cv_folds = cv_folds
)

# --- 12. Consolidate and Print Final Results ---

final_results_table <- bind_rows(
  analysis_output_ff5$results,
  analysis_output_ff3$results, 
  analysis_output_themes$results,
  analysis_output_factors$results,
  analysis_output_all$results
)

# The final table will now include tuned model names, Test_RMSE_pct and Test_MAE_pct
print(final_results_table, n = Inf)


#######################################################
### --- 13. Plot Test Set Predicted vs. Actual Returns --- ###
#######################################################

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
# (This section is unchanged)

# Plot 1: FF5 Only
plot_predicted_returns(
  predictions_tbl = analysis_output_ff5$predictions,
  analysis_label = "FF5 Predictors Only"
)

# Plot 2: FF3 Only
plot_predicted_returns(
  predictions_tbl = analysis_output_ff3$predictions,
  analysis_label = "FF3 Predictors Only"
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

# Plot 5: All Predictors
plot_predicted_returns(
  predictions_tbl = analysis_output_all$predictions,
  analysis_label = "All Predictors"
)


#################################################
### --- 14. Plot Test Set Cumulative Returns --- ###
#################################################

# --- Define a reusable plotting function ---
# (This function is unchanged)
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
        y = "Cumulative Return (1 + r)",
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
# (This section is unchanged)

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


#################################################
### --- 15. Plot Variable Importance --- ###
#################################################

# --- Define a reusable plotting function for variable importance ---
plot_variable_importance <- function(importance_df, model_title) {
  
  # Get top 15 variables
  top_15_vars <- importance_df %>%
    arrange(desc(Importance)) %>%
    slice_head(n = 15)
  
  # Create the plot
  print(
    ggplot(top_15_vars, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(
        title = paste("Top 15 Most Important Predictors", model_title),
        x = "Predictor",
        y = "Importance (IncNodePurity / Rel. Inf.)"
      ) +
      theme_minimal()
  )
}

# --- Plot importance for the "All Predictors" analysis ---
# This is likely your most interesting model set

# Random Forest Importance
plot_variable_importance(
  importance_df = analysis_output_all$importance_rf,
  model_title = "Random Forest (All Predictors)"
)

# Gradient Boosting Importance
plot_variable_importance(
  importance_df = analysis_output_all$importance_gbm,
  model_title = "Gradient Boosting (All Predictors)"
)