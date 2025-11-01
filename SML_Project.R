# --- 1. Load All Necessary Packages ---

all_packages <- c(
  "dplyr", 
  "frenchdata",
  "ggplot2",
  "lubridate",
  "tidyverse",
  "quantmod"
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

