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
  Model = c("OLS (Baseline)", "Elastic Net (Tuned)", "Random Forest"),
  Test_MSE = c(mse_ols, mse_enet, mse_rf)
)

print(results)
