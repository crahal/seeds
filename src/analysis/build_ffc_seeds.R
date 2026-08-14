## Replication code for the FFC benchmarks
## by Ian Lundberg. Modified by Charlie
## Rahal to analyze seed variability.
## ilundberg at princeton dot edu
## charles dot rahal at sociology .ac.uk

#x <- 08544 original seed

############################
## Load required packages ##
############################

library(tidyverse)
library(magrittr)
library(haven)
library(forcats)
library(reshape2)
library(foreach)
library(readstata13)
library(Amelia)
library(ranger)
library(quadprog)
library(readr)
library(here)
rm(list = ls())

private.data.dir <- file.path(getwd(), 'private')
output.dir <- file.path(getwd(), 'output')
imputed.dir <- file.path(output.dir, 'imputed')
benchmark.dir <- file.path(output.dir, 'benchmarks')
seed.dir <- file.path(output.dir, 'seed')

background <- read.dta13(file.path(private.data.dir, "background.dta"), convert.factors = F)
train <- read_csv(file.path(private.data.dir, "train.csv"))
test <- read_csv(file.path(private.data.dir, "test.csv"))
outcomes <- colnames(train)[-1]

max_seeds <- 1
for(x in 1:max_seeds) {
  cat("\r", x, "of", max_seeds, '! ') 
#  flush.console()
  x <- 08544# original seed
  set.seed(x)
  d <- background %>%
    mutate(cm1relf = ifelse(cm1relf == 1, "Married",
                            ifelse(cm1relf == 2, "Cohabiting",
                                   ifelse(cm1relf >= 3, "Other",NA))),
           cm1ethrace = ifelse(cm1ethrace %in% c(1,4), "White/other",
                               ifelse(cm1ethrace == 2, "Black",
                                      ifelse(cm1ethrace == 3, "Hispanic", NA))),
           cm1edu = factor(ifelse(cm1edu >= 1, cm1edu, NA),
                           labels = c("Less than high school",
                                      "High school",
                                      "Some college",
                                      "College")),
           gpa9 = 1/3 * (ifelse(t5c13a > 0, t5c13a, NA) +
                           ifelse(t5c13b > 0, t5c13b, NA) +
                           ifelse(t5c13c > 0, t5c13c, NA)),
           grit9 = 1/3 * (ifelse(t5b2b > 0, t5b2b, NA) +
                            ifelse(t5b4y >= 0, 4 - t5b4y, NA) +
                            ifelse(t5b4z >= 0, 4 - t5b4z, NA)),
           materialHardship9 = ifelse(
             m5a2 %in% c(1,2),
             1 / 10 * (
               ifelse(m5f23a > 0, m5f23a == 1, NA) +
                 ifelse(m5f23b > 0, m5f23b == 1, NA) +
                 ifelse(m5f23c > 0, m5f23c == 1, NA) +
                 ifelse(m5f23d > 0, m5f23d == 1, NA) +
                 ifelse(m5f23e > 0, m5f23e == 1, NA) +
                 ifelse(m5f23f > 0, m5f23f == 1, NA) +
                 ifelse(m5f23g > 0, m5f23g == 1, NA) +
                 ifelse(m5f23h > 0, m5f23h == 1, NA) +
                 ifelse(m5f23i > 0, m5f23i == 1, NA) +
                 ifelse(m5f23j > 0, m5f23j == 1, NA)
             ),
             ifelse(f5a2 %in% c(1,2),
                    1 / 10 * (
                      ifelse(f5f23a > 0, f5f23a == 1, NA) +
                        ifelse(f5f23b > 0, f5f23b == 1, NA) +
                        ifelse(f5f23c > 0, f5f23c == 1, NA) +
                        ifelse(f5f23d > 0, f5f23d == 1, NA) +
                        ifelse(f5f23e > 0, f5f23e == 1, NA) +
                        ifelse(f5f23f > 0, f5f23f == 1, NA) +
                        ifelse(f5f23g > 0, f5f23g == 1, NA) +
                        ifelse(f5f23h > 0, f5f23h == 1, NA) +
                        ifelse(f5f23i > 0, f5f23i == 1, NA) +
                        ifelse(f5f23j > 0, f5f23j == 1, NA)
                    ),
                    1 / 10 * (
                      ifelse(n5g1a > 0, n5g1a == 1, NA) +
                        ifelse(n5g1b > 0, n5g1b == 1, NA) +
                        ifelse(n5g1c > 0, n5g1c == 1, NA) +
                        ifelse(n5g1d > 0, n5g1d == 1, NA) +
                        ifelse(n5g1e > 0, n5g1e == 1, NA) +
                        ifelse(n5g1f > 0, n5g1f == 1, NA) +
                        ifelse(n5g1g > 0, n5g1g == 1, NA) +
                        ifelse(n5g1h > 0, n5g1h == 1, NA) +
                        ifelse(n5g1i > 0, n5g1i == 1, NA) +
                        ifelse(n5g1j > 0, n5g1j == 1, NA)
                    ))
           ),
           eviction9 = ifelse(m5a2 %in% c(1,2),
                              ifelse(m5f23d <= 0, NA, m5f23d == 1),
                              ifelse(f5a2 %in% c(1,2),
                                     ifelse(f5f23d <= 0, NA, f5f23d == 1),
                                     NA)),
           layoff9 = ifelse(m5a2 %in% c(1,2),
                            ifelse(m5i4 > 0, m5i4 == 2, NA),
                            ifelse(f5a2 %in% c(1,2),
                                   ifelse(f5i4 > 0, f5i4 == 2, NA),
                                   NA)),
           jobTraining9 = ifelse(m5a2 %in% c(1,2),
                                 ifelse(m5i3b > 0, m5i3b == 1, NA),
                                 ifelse(f5a2 %in% c(1,2),
                                        ifelse(f5i3b > 0, f5i3b == 1, NA),
                                        NA))) %>%
    select(challengeID, cm1ethrace, cm1relf, cm1edu,
           gpa9, grit9, materialHardship9, eviction9, layoff9, jobTraining9) %>%
    left_join(train, by = "challengeID")
  d[apply(d[,-1],1,function(x) all(is.na(x))),"cm1ethrace"] <- "White/other"

  get.benchmark.predictions <- function(outcome, model = "full", data = d) {
    if(model == "full") {
      # Create the formula dynamically
      thisFormula <- formula(paste0(outcome, " ~ cm1ethrace + cm1relf + cm1edu + ", outcome, "9"))
      
      # Perform the imputation using Amelia
      imputed <- amelia(data %>% select(challengeID, cm1ethrace, cm1relf, cm1edu, contains(outcome)),
                        m = 1,
                        p2s = 0,
                        noms = c("cm1ethrace", "cm1relf"),
                        ords = "cm1edu",
                        idvars = "challengeID")$imputations$imp1
      
      # Check if the required outcome column exists in the imputed data
      required_column <- paste0(outcome, "9")
      if (!required_column %in% colnames(imputed)) {
        stop(paste("Column", required_column, "does not exist in imputed data"))
      }
      
      # Identify rows where all predictors are missing
      missing_all_predictors <- apply(get_all_vars(thisFormula, data = imputed), 1, function(x) all(is.na(x[-1])))
      ols.yhat <- logit.yhat <- rep(NA, nrow(imputed))
      ols.yhat[missing_all_predictors] <- logit.yhat[missing_all_predictors] <- mean(imputed[,outcome], na.rm = TRUE)
      
      # Fit the OLS model
      ols <- tryCatch(
        lm(formula = thisFormula, data = imputed[!is.na(imputed[,outcome]), ]),
        warning = function(w) {
          message("Warning in lm: ", w)
          return(NULL)
        },
        error = function(e) {
          message("Error in lm: ", e)
          return(NULL)
        }
      )
      
      # Extract the coefficient for the outcome predictor
      if (!is.null(ols)) {
        ols_summary <- coef(summary(ols))
        if (paste0(outcome, "9") %in% rownames(ols_summary)) {
          ols_beta <- ols_summary[paste0(outcome, "9"), "Estimate"]
        } else {
          message(paste("Coefficient for", paste0(outcome, "9"), "not found in the model"))
          ols_beta <- NA
        }
      } else {
        ols_beta <- NA
      }
      
      # Fit a logistic regression model if the outcome is binary
      if (length(unique(na.omit(imputed[, outcome]))) == 2) {
        logit <- tryCatch(
          glm(formula = thisFormula, family = binomial(link = "logit"), data = imputed[!is.na(imputed[,outcome]), ]),
          warning = function(w) {
            message("Warning in glm: ", w)
            return(NULL)
          },
          error = function(e) {
            message("Error in glm: ", e)
            return(NULL)
          }
        )
        
        if (!is.null(logit)) {
          logit_summary <- coef(summary(logit))
          print(logit_summary)
          if (paste0(outcome, "9") %in% rownames(logit_summary)) {
            logit.yhat[!missing_all_predictors] <- predict(logit, newdata = imputed[!missing_all_predictors,], type = "response")
            logit_beta <- logit_summary[paste0(outcome, "9"), "Estimate"]
          } else {
            message(paste("Coefficient for", paste0(outcome, "9"), "not found in the logistic model"))
            logit_beta <- NA
          }
        } else {
          logit.yhat <- rep(NA, nrow(imputed))
          logit_beta <- NA
        }
      } else {
        logit.yhat <- rep(NA, nrow(imputed))
        logit_beta <- NA
      }
      
      # Create a data frame with all predictions and relevant information
      all_predictions <- data.frame(outcome = outcome,
                                    challengeID = imputed$challengeID,
                                    ols = ols.yhat,
                                    ols_beta = ols_beta,
                                    logit = logit.yhat,
                                    logit_beta = logit_beta) %>%
        mutate(ols = case_when(outcome %in% c("grit", "gpa") & ols < 1 ~ 1,
                               outcome %in% c("grit", "gpa") & ols > 4 ~ 4,
                               outcome %in% c("grit", "gpa") ~ ols,
                               ols < 0 ~ 0,
                               ols > 1 ~ 1,
                               TRUE ~ ols),
               logit = case_when(logit < 0 ~ 0,
                                 logit > 1 ~ 1,
                                 TRUE ~ as.numeric(logit)))
      
      return(all_predictions)
    }
  }
  
  benchmarks <- data.frame() 

  for (thisOutcome in outcomes) {
    for (predictor_set in c("full")) {
      print(thisOutcome)
      
      result <- get.benchmark.predictions(thisOutcome, model = predictor_set) %>%
        mutate(predictors = predictor_set)
      
      # Append the result to the benchmarks data frame
      benchmarks <- rbind(benchmarks, result)
    }
  }
  
  print('here?')
  benchmarks_long <- benchmarks %>%
    select(challengeID, outcome, ols, logit,# rf,
           predictors) %>%
    melt(id = c("challengeID", "outcome", "predictors"),
         variable.name = "account",
         value.name = "prediction") %>%
    mutate(account = paste(account)) %>%
    select(-predictors) %>%
    right_join(
      test %>%
        melt(id = "challengeID", variable.name = "outcome", value.name = "truth") %>%
        select(challengeID, outcome, truth),
      by = c("challengeID","outcome")
    ) %>%
    left_join(
      train %>%
        melt(id = "challengeID", variable.name = "outcome") %>%
        group_by(outcome) %>%
        summarize(ybar_train = mean(value, na.rm = T)),
      by = c("outcome")
    ) %>%
    group_by(outcome, account) %>%
    mutate(r2_holdout = 1 - mean((truth - prediction) ^ 2, na.rm = T) / mean((truth - ybar_train) ^ 2, na.rm = T),
           beatingBaseline = r2_holdout > 0,
           outcome_name = case_when(outcome == "materialHardship" ~ "A. Material\nhardship",
                                    outcome == "gpa" ~ "B. GPA",
                                    outcome == "grit" ~ "C. Grit",
                                    outcome == "eviction" ~ "D. Eviction",
                                    outcome == "jobTraining" ~ "E. Job\ntraining",
                                    outcome == "layoff" ~ "F. Layoff")) %>%
     select(outcome, account, r2_holdout) %>%
     arrange(outcome, account )
  benchmarks_long <- benchmarks_long[!duplicated(benchmarks_long), ]
  
  
  benchmark_beta <- benchmarks[c("outcome", "ols_beta", "logit_beta")][!duplicated(benchmarks[c("outcome", "ols_beta", "logit_beta")]), ]
  benchmark_beta_melted <- melt(benchmark_beta, variable.name="account", value.name="beta")
  benchmark_beta_melted <- as.data.frame(sapply(benchmark_beta_melted,gsub,pattern="_beta",replacement=""))
  if (exists('out')){
    out_new <- merge(benchmarks_long, benchmark_beta_melted, by=c("outcome","account")) # NA's match
    out_new['seed'] <- x
    out <- rbind(out, out_new)
  } else{
    out <- merge(benchmarks_long, benchmark_beta_melted, by=c("outcome","account")) # NA's match
    out['seed'] <- x
  }
}

write_csv(
  out,
  path = file.path(seed.dir, sprintf("seed_analysis_%s.csv", max_seeds))
)



