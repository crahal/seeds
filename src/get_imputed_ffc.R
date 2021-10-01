#####################################
## Plots for:                      ##
## Measuring the Predictability of ##
## Life Outcomes with a Scientific ##
## Mass Collaboration              ##
#####################################

## Code by Ian Lundberg
## ilundberg at princeton dot edu

#####################
## Set seed for    ##
## reproducibility ##
#####################
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
# Set directory information

# Set ggplot2 theme
setwd('Dropbox/seeds/data/ffc/')
os = 'linux'
private.data.dir <- file.path(getwd(), 'private')
output.dir <- file.path(getwd(), 'output', os)
imputed.dir <- file.path(output.dir, 'imputed')
benchmark.dir <- file.path(output.dir, 'benchmarks')
seed.dir <- file.path(output.dir, 'seed')

background <- read.dta13(file.path(private.data.dir, "background.dta"), convert.factors = F)
train <- read_csv(file.path(private.data.dir, "train.csv"))
test <- read_csv(file.path(private.data.dir, "test.csv"))
outcomes <- colnames(train)[-1]
###############################
## Benchmarks for supplement ##
###############################

max_seeds <- 1000000
for(x in 1:max_seeds) {
  Sys.sleep(.1)
  cat("\r", x, "of", max_seeds, '! ') 
  flush.console()
  
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
           ## For prior measure of GPA, use the teacher report of skills
           ## in language and literacy,
           ## in science and social studies,
           ## and in math,
           ## all coded 1 = far below average to 5 = far above average
           gpa9 = 1/3 * (ifelse(t5c13a > 0, t5c13a, NA) +
                           ifelse(t5c13b > 0, t5c13b, NA) +
                           ifelse(t5c13c > 0, t5c13c, NA)),
           ## For grit, use teacher reports of:
           ## Child persists in completing tasks
           ## Child fails to finish things he or she starts (reverse coded)
           ## Child does not follow through on instructions and fails to finish homework
           ## EXCLUDE Child pays attention well
           ## EXCLUDE Child ignores peer distractions when doing class work
           ## EXCLUDE Child has a short attention span
           ## EXCLUDE Child has distractibility or attention span problem
           grit9 = 1/3 * (ifelse(t5b2b > 0, t5b2b, NA) +
                            ifelse(t5b4y >= 0, 4 - t5b4y, NA) +
                            ifelse(t5b4z >= 0, 4 - t5b4z, NA)),
           materialHardship9 = ifelse(
             m5a2 %in% c(1,2),
             ## Mother's material hardship
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
                    ## Father's material hardship
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
                    ## PCG material hardship
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
           ## Use whether did work for pay the week of the age 9 interview
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
  
  ## For one row that is missing everything, fill in that race is white/other
  ## so we can impute everything from there. This case is likely missing
  ## in the test set and has no training outcomes, so this won't matter.
  
  d[apply(d[,-1],1,function(x) all(is.na(x))),"cm1ethrace"] <- "White/other"
  
  ####################
  ## Fit benchmarks ##
  ####################
  
  ## Function to make OLS predictions
  get.benchmark.predictions <- function(outcome, model = "full", data = d) {
    if(model == "full") {
      thisFormula <- formula(paste0(outcome,
                                    " ~ cm1ethrace + cm1relf + cm1edu + ",
                                    outcome,"9"))
      imputed <- amelia(data %>% select(challengeID, cm1ethrace, cm1relf, cm1edu, contains(outcome)),
                        m = 1,
                        p2s = 0,
                        noms = c("cm1ethrace","cm1relf"),
                        ords = "cm1edu",
                        idvars = "challengeID")$imputations$imp1
    }
    
    # Identify the rows that are missing all predictors
    # This happens if there were no variables with valid values,
    # giving Amelia no data with which to impute.
    missing_all_predictors <- apply(get_all_vars(thisFormula, data = imputed), 1, function(x) all(is.na(x[-1])))
    # Create holders for the predicted values from the models
    ols.yhat <- logit.yhat <- rf.yhat <- rep(NA, nrow(imputed))
    
    # If missing all predictors, impute the grand mean
    ols.yhat[missing_all_predictors] <- 
      logit.yhat[missing_all_predictors] <- 
      rf.yhat[missing_all_predictors] <- 
      mean(imputed[,outcome], na.rm = T)
    
    # Fit models to impute predictions when predictors are available
    
    # OLS
    ols <- lm(formula = thisFormula,
              data = imputed[!is.na(data[,outcome]),])
    
    
    filename = sprintf("imputed_benchmark_%s.csv", outcome)
    form = here(imputed.dir, filename)
    write.csv(imputed %>% select(challengeID, cm1ethrace, cm1relf, cm1edu, paste0(outcome,9)),
              file=form, row.names = FALSE)
    
    
    ols.yhat[!missing_all_predictors] <- predict(ols, newdata = imputed[!missing_all_predictors,])
    ols.beta <- coef(summary(ols))[paste0(outcome,9),"Estimate"]
    # Logit for binary outcomes
    if (length(unique(na.omit(data[,outcome]))) == 2) {
      logit <- glm(formula = thisFormula,
                   family = binomial(link = "logit"),
                   data = imputed[!is.na(data[,outcome]),])
      logit.yhat[!missing_all_predictors] <- predict(ols, newdata = imputed[!missing_all_predictors,], type = "response")
      logit.beta <- coef(summary(logit))[paste0(outcome,9),"Estimate"]
    } else {
      # If not binary, make all logit predictions NA
      logit.yhat <- NA
      logit.beta <- NA
    }
    
    # Random forest
    rf <- ranger(thisFormula,
                 data = imputed[!is.na(data[,outcome]),])
    rf.yhat[!missing_all_predictors] <- predict(rf, data = imputed[!missing_all_predictors,])$predictions
    
    # Combine into one data frame
    # and truncate to observable range
    all_predictions <- data.frame(outcome = outcome,
                                  challengeID = imputed$challengeID,
                                  ols = ols.yhat,
                                  ols_beta = ols.beta,
                                  logit = logit.yhat,
                                  logit_beta = logit.beta,
                                  rf = rf.yhat) %>%
      mutate(ols = case_when(outcome %in% c("grit","gpa") & ols < 1 ~ 1,
                             outcome %in% c("grit","gpa") & ols > 4 ~ 4,
                             outcome %in% c("grit","gpa") ~ ols,
                             ols < 0 ~ 0,
                             ols > 1 ~ 1,
                             T ~ ols),
             logit = case_when(logit < 0 ~ 0,
                               logit > 1 ~ 1,
                               T ~ as.numeric(logit)),
             rf = case_when(outcome %in% c("grit","gpa") & rf < 1 ~ 1,
                            outcome %in% c("grit","gpa") & rf > 4 ~ 4,
                            outcome %in% c("grit","gpa") ~ rf,
                            rf < 0 ~ 0,
                            rf > 1 ~ 1,
                            T ~ rf))
    return(all_predictions)
  }
  
  # Get benchmarks on all outcomes
  benchmarks <- foreach(thisOutcome = outcomes, .combine = "rbind") %do% {
    foreach(predictor_set = c("full"), .combine = "rbind") %do% {
      get.benchmark.predictions(thisOutcome, model = predictor_set) %>%
        mutate(predictors = predictor_set)
    }
  }
  
  # Output a version stored like submissions.csv
  # Matt will use this in his figures
  benchmarks_long <- benchmarks %>%
    select(challengeID, outcome, ols, logit, rf, predictors) %>%
    melt(id = c("challengeID", "outcome", "predictors"),
         variable.name = "account",
         value.name = "prediction") %>%
    mutate(account = paste(account)) %>%
    select(-predictors) %>%
    # Add information so that this is formatted like submissions.csv
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
  #  select(outcome, outcome_name, account, challengeID, prediction, truth, ybar_train, r2_holdout, beatingBaseline) %>%
  #  arrange(outcome_name, account, challengeID)
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
  path = file.path(seed.dir, "seed_analysis.csv")
)
