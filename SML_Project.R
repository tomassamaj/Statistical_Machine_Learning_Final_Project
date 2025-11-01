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
  "ranger"
)

options(repos = "https://cloud.r-project.org")

installed <- rownames(installed.packages())
for(pkg in all_packages) {
  if(! pkg %in% installed) install.packages(pkg)
}

invisible(lapply(all_packages, library, character.only = TRUE))

# --- 2. Set Working Directory ---
print(getwd())

#########################################
### Load and Prepare Fama-French Data ###
#########################################

start_date <- ymd("1993-01-01") 
end_date <- ymd("2024-12-31")

factors_ff5_monthly_raw <- download_french_data("Fama/French 5 Factors (2x3)")

factors_ff5_monthly <- factors_ff5_monthly_raw$subsets$data[[1]] |>
 mutate(
    # Use rollforward() to get the last day of the month
  date = rollforward(ymd(paste0(date, "01"))), 
  across(c(RF, `Mkt-RF`, SMB, HML, RMW, CMA), ~as.numeric(.) / 100),
  .keep = "none"
 ) |>
 rename_with(str_to_lower) |>
 rename(mkt_excess = `mkt-rf`) |>
 filter(date >= start_date & date <= end_date)

# Drop the first row
factors_ff5_monthly <- factors_ff5_monthly[-1,]

########################################################################
### Download monthly SP500 returns from yahoo finance using quantmod ###
########################################################################

# getSymbols will fetch all available data within your specified date range.
sp500_raw <- getSymbols("SPY", 
                        src = "yahoo", 
                        from = start_date, 
                        to = end_date, 
                        auto.assign = FALSE)

# Calculate monthly returns from the Adjusted price column (Ad())
# Using the Adjusted price is crucial as it accounts for dividends and splits.
sp500_monthly_returns <- monthlyReturn(Ad(sp500_raw))

# Convert the xts object to a tibble for easier manipulation and merging
# with your Fama-French tibble from the previous step.
sp500_returns_tbl <- sp500_monthly_returns |>
  as_tibble(rownames = "date") |>
  mutate(date = ymd(date)) |>
  rename(sp500_return = monthly.returns)

# Drop the first row
sp500_returns_tbl <- sp500_returns_tbl[-1,]

##########################################################################
### Load monthly factor and theme data from the Global factor database ###
##########################################################################

# load csv file with factor returns

all_factors_monthly_vw_cap  <- read.csv("[usa]_[all_factors]_[monthly]_[vw_cap].csv")
all_themes_monthly_vw_cap  <- read.csv("[usa]_[all_themes]_[monthly]_[vw_cap].csv")

# make them from long format to wide with dplyer through the name col

# drop all other cols except name, date and ret
all_factors_monthly_vw_cap <- all_factors_monthly_vw_cap  %>%
  select(date, name, ret)

all_factors_wide <- all_factors_monthly_vw_cap %>%
  pivot_wider(names_from = name, values_from = ret)

all_themes_monthly_vw_cap  <- all_themes_monthly_vw_cap  %>%
  select(date, name, ret)

all_themes_wide <- all_themes_monthly_vw_cap  %>%
  pivot_wider(names_from = name, values_from = ret)

# set date cols to date
all_factors_wide$date <- as.Date(as.character(all_factors_wide$date))
all_themes_wide$date <- as.Date(as.character(all_themes_wide$date))

# filter dates
all_factors_wide <- all_factors_wide %>%
  filter(date >= start_date & date <= end_date)
all_themes_wide <- all_themes_wide %>%
  filter(date >= start_date & date <= end_date)

str(all_factors_wide)
str(all_themes_wide)

# drop the 1st row
all_factors_wide <- all_factors_wide[-1,]
all_themes_wide <- all_themes_wide[-1,]

################################################################################
### Get Colnames For Different Predictor Sets (FF5, All Themes, All Factors) ###
################################################################################

# Get theme names for the second analysis (Analysis 2)
# We get the column names *before* the merge, excluding 'date'
theme_names <- all_themes_wide %>% 
  select(-date) %>% 
  colnames()

# Add the '_lag1' suffix to match the predictor names in 'final_data'
theme_predictors_lagged <- paste0(theme_names, "_lag1")

print("Theme predictors for Analysis 2 identified:")
print(theme_predictors_lagged)

# Get factor names for the second analysis (Analysis 3)
# We get the column names *before* the merge, excluding 'date'
factor_names <- all_factors_wide %>% 
  select(-date) %>% 
  colnames()

# Add the '_lag1' suffix to match the predictor names in 'final_data'
factor_predictors_lagged <- paste0(factor_names, "_lag1")

print("Factor predictors for Analysis 3 identified:")
print(factor_predictors_lagged)


###########################################
### Merge All Data into a Master Tibble ###
###########################################

# Join all datasets. An inner_join ensures we only keep
# dates where we have data for all sources.
master_tbl <- sp500_returns_tbl |>
  inner_join(factors_ff5_monthly, by = "date") |>
  inner_join(all_factors_wide, by = "date") |>
  inner_join(all_themes_wide, by = "date") |>
  
  # Calculate your target variable (Y): SP500 Excess Return
  mutate(sp500_excess = sp500_return - rf) |>
  
  # Select and arrange columns for clarity
  select(date, sp500_excess, mkt_excess, smb, hml, rmw, cma, rf, everything(), -sp500_return)

head(master_tbl)

# Create Predictive Lagged Dataset ---

# We are predicting Y(t) using X(t-1)
# Y = sp500_excess at time t
# X = all other factors from time t-1
final_data <- master_tbl |>
  # Lag all columns EXCEPT the date and the target (Y)
  mutate(across(-c(date, sp500_excess), lag, .names = "{.col}_lag1")) |>  
  # Keep only the target and the new lagged predictors
  select(date, sp500_excess, ends_with("_lag1")) |>
  # The first row will have NAs from lagging, so we must remove it
  slice(-1)
# This is the final, analysis-ready dataset
head(final_data)

# --- 7. Create Time-Series Splits (Train/Val/Test) ---

# Use a 50% train, 25% validation, 25% test split
n <- nrow(final_data)
train_end <- floor(0.50 * n)
val_end <- floor(0.75 * n)

train_data <- final_data |> slice(1:train_end)
val_data   <- final_data |> slice((train_end + 1):val_end)
test_data  <- final_data |> slice((val_end + 1):n)

# --- 8. Pre-processing: Standardize the Data ---

# IMPORTANT: You must learn the scaling parameters (mean, sd)
# ONLY from the training data, and then apply them to all three sets.
# This prevents data leakage.

# 1. Get scaling parameters from training data
# We ignore date (col 1) and target (col 2)
preproc_params <- preProcess(train_data[, -c(1, 2)], 
                             method = c("center", "scale"))

# 2. Transform all three datasets
train_data_proc <- predict(preproc_params, train_data)
val_data_proc   <- predict(preproc_params, val_data)
test_data_proc  <- predict(preproc_params, test_data)

# You will also need X and Y matrices for glmnet
# Training set
x_train <- as.matrix(train_data_proc[, -c(1, 2)])
y_train <- train_data_proc$sp500_excess

# Validation set
x_val <- as.matrix(val_data_proc[, -c(1, 2)])
y_val <- val_data_proc$sp500_excess

# Test set
x_test <- as.matrix(test_data_proc[, -c(1, 2)])
y_test <- test_data_proc$sp500_excess

### Fit OLS on the TEST Data ###
# Fit on TRAINING data only
ols_model <- lm(sp500_excess ~ ., data = train_data_proc |> select(-date))
summary(ols_model)

### Elastic Net ###
# --- Hyperparameter Tuning for Elastic Net ---

# We will test alpha = 0 (Ridge), 0.5 (Elastic Net), 1 (Lasso)
# You can try a finer grid later
alphas_to_try <- c(0, 0.5, 1) 
best_mse <- Inf
best_model <- NULL

for (a in alphas_to_try) {
  # 1. Fit the model on TRAINING data
  # cv.glmnet finds the best lambda
  cv_fit <- cv.glmnet(x_train, y_train, alpha = a, family = "gaussian")
  
  # 2. Predict on VALIDATION data using the best lambda (lambda.min)
  preds_val <- predict(cv_fit, newx = x_val, s = "lambda.min")
  
  # 3. Calculate MSE on VALIDATION data
  mse_val <- mean((y_val - preds_val)^2)
  
  # 4. Check if this model is the best one so far
  if (mse_val < best_mse) {
    best_mse   <- mse_val
    best_model <- cv_fit
  }
}

print(paste("Best Alpha (from validation):", best_model$glmnet.fit$alpha))
print(paste("Best Lambda (from validation):", best_model$lambda.min))

### Random Forest ###
# Fit on TRAINING data
rf_model <- ranger(
  formula   = sp500_excess ~ ., 
  data      = train_data_proc |> select(-date),
  num.trees = 500,
  mtry      = floor(ncol(x_train) / 3) # A common heuristic for mtry
)

# --- 9. Final Model Assessment on TEST Data ---

# 1. OLS Predictions
preds_ols <- predict(ols_model, newdata = test_data_proc)
mse_ols <- mean((y_test - preds_ols)^2)

# 2. Elastic Net Predictions (using the best model found in Step 4)
preds_enet <- predict(best_model, newx = x_test, s = "lambda.min")
mse_enet <- mean((y_test - preds_enet)^2)

# 3. Random Forest Predictions
preds_rf <- predict(rf_model, data = test_data_proc)$predictions
mse_rf <- mean((y_test - preds_rf)^2)

# --- 10. Compare Final Performance ---
results <- tibble(
 Model = c("OLS (All Predictors)", "Elastic Net (All Predictors)", "Random Forest (All Predictors)"),
 Test_MSE = c(mse_ols, mse_enet, mse_rf)
)

print("--- ANALYSIS 1: ALL PREDICTORS ---")
print(results)


###################################################
### --- ANALYSIS 2: THEME-ONLY PREDICTORS --- ###
###################################################

# --- 9-B. Create Theme-Only Data Subsets ---

# Create data frames for lm() and ranger() formula interface
# We select the target variable + our list of theme predictors
train_data_themes <- train_data_proc %>% 
  select(sp500_excess, all_of(theme_predictors_lagged))
test_data_themes <- test_data_proc %>% 
  select(sp500_excess, all_of(theme_predictors_lagged))
  
# Create matrices for glmnet() by subsetting the original matrices
# We use 'drop = FALSE' to ensure it stays a matrix even if there's one predictor
x_train_themes <- x_train[, theme_predictors_lagged, drop = FALSE]
x_val_themes   <- x_val[, theme_predictors_lagged, drop = FALSE]
x_test_themes  <- x_test[, theme_predictors_lagged, drop = FALSE]

# The Y variables (y_train, y_val, y_test) are the same


# --- 10-B. Fit Theme-Only Models ---

### Fit OLS on the THEME Data ###
ols_model_themes <- lm(sp500_excess ~ ., data = train_data_themes)

### Tune Elastic Net on THEME Data ###
alphas_to_try <- c(0, 0.5, 1) 
best_mse_themes <- Inf
best_model_themes <- NULL

for (a in alphas_to_try) {
  # 1. Fit on TRAINING data
  cv_fit_themes <- cv.glmnet(x_train_themes, y_train, alpha = a, family = "gaussian")
  
  # 2. Predict on VALIDATION data
  preds_val_themes <- predict(cv_fit_themes, newx = x_val_themes, s = "lambda.min")
  
  # 3. Calculate MSE on VALIDATION data
  mse_val_themes <- mean((y_val - preds_val_themes)^2)
  
  # 4. Keep the best model
  if (mse_val_themes < best_mse_themes) {
    best_mse_themes   <- mse_val_themes
    best_model_themes <- cv_fit_themes
  }
}

print(paste("Best Alpha (Themes):", best_model_themes$glmnet.fit$alpha))
print(paste("Best Lambda (Themes):", best_model_themes$lambda.min))

### Fit Random Forest on THEME Data ###
rf_model_themes <- ranger(
  formula   = sp500_excess ~ ., 
  data      = train_data_themes,
  num.trees = 500,
  # Adjust mtry based on the new, smaller number of predictors
  mtry      = floor(ncol(x_train_themes) / 3) 
)

# --- 11-B. Assess Theme-Only Models on TEST Data ---

# 1. OLS Predictions (Themes)
preds_ols_themes <- predict(ols_model_themes, newdata = test_data_themes)
mse_ols_themes <- mean((y_test - preds_ols_themes)^2)

# 2. Elastic Net Predictions (Themes)
preds_enet_themes <- predict(best_model_themes, newx = x_test_themes, s = "lambda.min")
mse_enet_themes <- mean((y_test - preds_enet_themes)^2)

# 3. Random Forest Predictions (Themes)
preds_rf_themes <- predict(rf_model_themes, data = test_data_themes)$predictions
mse_rf_themes <- mean((y_test - preds_rf_themes)^2)

# --- 12-B. Compare Final Performance (Themes) ---
results_themes <- tibble(
  Model = c("OLS (Themes)", "Elastic Net (Themes)", "Random Forest (Themes)"),
  Test_MSE = c(mse_ols_themes, mse_enet_themes, mse_rf_themes)
)

print("--- ANALYSIS 2: THEME-ONLY PREDICTORS ---")
print(results_themes)


##################################################
### --- ANALYSIS 3: ALL-FACTORS PREDICTORS --- ###
##################################################

# --- 9-B. Create Factor-Only Data Subsets ---

# Create data frames for lm() and ranger() formula interface
# We select the target variable + our list of theme predictors
train_data_factors <- train_data_proc %>% 
  select(sp500_excess, all_of(factor_predictors_lagged))
test_data_factors <- test_data_proc %>% 
  select(sp500_excess, all_of(factor_predictors_lagged))
  
# Create matrices for glmnet() by subsetting the original matrices
# We use 'drop = FALSE' to ensure it stays a matrix even if there's one predictor
x_train_factors <- x_train[, theme_predictors_lagged, drop = FALSE]
x_val_factors   <- x_val[, theme_predictors_lagged, drop = FALSE]
x_test_themes  <- x_test[, theme_predictors_lagged, drop = FALSE]

# The Y variables (y_train, y_val, y_test) are the same


# --- 10-B. Fit Theme-Only Models ---

### Fit OLS on the THEME Data ###
ols_model_themes <- lm(sp500_excess ~ ., data = train_data_themes)

### Tune Elastic Net on THEME Data ###
alphas_to_try <- c(0, 0.5, 1) 
best_mse_themes <- Inf
best_model_themes <- NULL

for (a in alphas_to_try) {
  # 1. Fit on TRAINING data
  cv_fit_themes <- cv.glmnet(x_train_themes, y_train, alpha = a, family = "gaussian")
  
  # 2. Predict on VALIDATION data
  preds_val_themes <- predict(cv_fit_themes, newx = x_val_themes, s = "lambda.min")
  
  # 3. Calculate MSE on VALIDATION data
  mse_val_themes <- mean((y_val - preds_val_themes)^2)
  
  # 4. Keep the best model
  if (mse_val_themes < best_mse_themes) {
    best_mse_themes   <- mse_val_themes
    best_model_themes <- cv_fit_themes
  }
}

print(paste("Best Alpha (Themes):", best_model_themes$glmnet.fit$alpha))
print(paste("Best Lambda (Themes):", best_model_themes$lambda.min))

### Fit Random Forest on THEME Data ###
rf_model_themes <- ranger(
  formula   = sp500_excess ~ ., 
  data      = train_data_themes,
  num.trees = 500,
  # Adjust mtry based on the new, smaller number of predictors
  mtry      = floor(ncol(x_train_themes) / 3) 
)

# --- 11-B. Assess Theme-Only Models on TEST Data ---

# 1. OLS Predictions (Themes)
preds_ols_themes <- predict(ols_model_themes, newdata = test_data_themes)
mse_ols_themes <- mean((y_test - preds_ols_themes)^2)

# 2. Elastic Net Predictions (Themes)
preds_enet_themes <- predict(best_model_themes, newx = x_test_themes, s = "lambda.min")
mse_enet_themes <- mean((y_test - preds_enet_themes)^2)

# 3. Random Forest Predictions (Themes)
preds_rf_themes <- predict(rf_model_themes, data = test_data_themes)$predictions
mse_rf_themes <- mean((y_test - preds_rf_themes)^2)

# --- 12-B. Compare Final Performance (Themes) ---
results_themes <- tibble(
  Model = c("OLS (Themes)", "Elastic Net (Themes)", "Random Forest (Themes)"),
  Test_MSE = c(mse_ols_themes, mse_enet_themes, mse_rf_themes)
)

print("--- ANALYSIS 2: THEME-ONLY PREDICTORS ---")
print(results_themes)



#####################################################
### --- ANALYSIS 3: FACTOR-ONLY PREDICTORS --- ###
#####################################################

# --- 9-C. Create Factor-Only Data Subsets ---

# Create data frames for lm() and ranger() formula interface
# We select the target variable + our list of factor predictors
train_data_factors <- train_data_proc %>% 
  select(sp500_excess, all_of(factor_predictors_lagged))
test_data_factors <- test_data_proc %>% 
  select(sp500_excess, all_of(factor_predictors_lagged))
  
# Create matrices for glmnet() by subsetting the original matrices
x_train_factors <- x_train[, factor_predictors_lagged, drop = FALSE]
x_val_factors   <- x_val[, factor_predictors_lagged, drop = FALSE]
x_test_factors  <- x_test[, factor_predictors_lagged, drop = FALSE]

# The Y variables (y_train, y_val, y_test) are the same


# --- 10-C. Fit Factor-Only Models ---

### Fit OLS on the FACTOR Data ###
ols_model_factors <- lm(sp500_excess ~ ., data = train_data_factors)

### Tune Elastic Net on FACTOR Data ###
alphas_to_try <- c(0, 0.5, 1) 
best_mse_factors <- Inf
best_model_factors <- NULL

for (a in alphas_to_try) {
  # 1. Fit on TRAINING data
  cv_fit_factors <- cv.glmnet(x_train_factors, y_train, alpha = a, family = "gaussian")
  
  # 2. Predict on VALIDATION data
  preds_val_factors <- predict(cv_fit_factors, newx = x_val_factors, s = "lambda.min")
  
  # 3. Calculate MSE on VALIDATION data
  mse_val_factors <- mean((y_val - preds_val_factors)^2)
  
  # 4. Keep the best model
  if (mse_val_factors < best_mse_factors) {
    best_mse_factors   <- mse_val_factors
    best_model_factors <- cv_fit_factors
  }
}

print(paste("Best Alpha (Factors):", best_model_factors$glmnet.fit$alpha))
print(paste("Best Lambda (Factors):", best_model_factors$lambda.min))

### Fit Random Forest on FACTOR Data ###
rf_model_factors <- ranger(
  formula   = sp500_excess ~ ., 
  data      = train_data_factors,
  num.trees = 500,
  # Adjust mtry based on the new, smaller number of predictors
  mtry      = floor(ncol(x_train_factors) / 3) 
)

# --- 11-C. Assess Factor-Only Models on TEST Data ---

# 1. OLS Predictions (Factors)
preds_ols_factors <- predict(ols_model_factors, newdata = test_data_factors)
mse_ols_factors <- mean((y_test - preds_ols_factors)^2)

# 2. Elastic Net Predictions (Factors)
preds_enet_factors <- predict(best_model_factors, newx = x_test_factors, s = "lambda.min")
mse_enet_factors <- mean((y_test - preds_enet_factors)^2)

# 3. Random Forest Predictions (Factors)
preds_rf_factors <- predict(rf_model_factors, data = test_data_factors)$predictions
mse_rf_factors <- mean((y_test - preds_rf_factors)^2)

# --- 12-C. Compare Final Performance (Factors) ---
results_factors <- tibble(
  Model = c("OLS (Factors)", "Elastic Net (Factors)", "Random Forest (Factors)"),
  Test_MSE = c(mse_ols_factors, mse_enet_factors, mse_rf_factors)
)

print("--- ANALYSIS 3: FACTOR-ONLY PREDICTORS ---")
print(results_factors)
