# Original script Written by Giacomo Vagni (https://github.com/giacomovagni)
# based on the original Vagni and Breen (2021) paper.

# Moduarlization undertaken by Daniel Valdenegro (https://github.com/dhvalden)
# Modifications for seeds by Charlie Rahal (https://github.com/crahal)
# Provided under its original CC BY-NC-SA license
# Last updated: 14/10/2022

library(Synth)
library(MSCMT)
library(parallel)
library(tidyverse)
library(data.table)
library(here)
options(dplyr.summarise.inform = FALSE)

load(here("data", "vagni_breen", "input", "isc_sample"))

fsample_person_period = function(pdf_prep, pid_treated_df, k_controls, seed){
  set.seed(seed)
  sample_control = pdf_prep %>%
        filter(treated == 0) %>%
        select(pidp) %>%
        distinct %>%
        sample_n(k_controls, replace = T)
  sample_control$pid_sampled = 2:(nrow(sample_control)+1)
  samp = rbind(pid_treated_df, sample_control)
  df_treat_sampled = suppressMessages(left_join(samp, pdf_prep))
  df_treat_sampled$pid_char = as.character(df_treat_sampled$pid_sampled)
  return(df_treat_sampled)
}

f_summarise_controls = function(x){
    x %>%
    group_by(timing_new, n_controls) %>%
    summarise(sum_zero_wt = sum(mscmt_w == 0),
      earnings_unwt = mean(earnings, na.rm = T),
      earnings_synth = earnings %*% mscmt_w,
      age_unwt = mean(age, na.rm = T),
      age_synth = age %*% mscmt_w,
      age_left_school_unwt = mean(age_left_school, na.rm = T),
      age_left_school_synth = age_left_school %*% mscmt_w,
      year_unwt = mean(year, na.rm = T),
      year_synth = year %*% mscmt_w)
}

l = length(isc_sample)
run <- function(seed, r){
    synth_w_df = list()
    synth_obj_save = list()
    for(i in 1:2){ #l?
        tryCatch({
            pdf_prep = isc_sample[[i]]
            pdf_prep$pid_char = as.character(pdf_prep$pidp)
            pdf_prep$timing_new2 = pdf_prep$timing_new + 1500 # 1500 random int, replace by argument of randomint func
            u = unique(pdf_prep$timing_new2[pdf_prep$treatment_period == 0 & pdf_prep$treated == 1] )
            p_id = unique(pdf_prep$pidp[pdf_prep$treated == 1] )
            c_id = unique(pdf_prep$pidp[pdf_prep$treated == 0] )
            pid_treated_df = data.frame(pidp = unique( pdf_prep$pidp[pdf_prep$treated == 1]) )
            pid_treated_df$pid_sampled = 1
            control_id = pdf_prep$pidp[pdf_prep$treated == 0]
            control_id = unique(control_id)
            k_controls = length(control_id)
            g = lapply(1:r, function(p) fsample_person_period(pdf_prep,
                                                              pid_treated_df = pid_treated_df,
                                                              k_controls = k_controls, seed)  )
            pdf_mscmt = lapply(1:r, function(i) g[[i]] [,c('pid_char',
                                                           'pid_sampled',
                                                           'earnings',
                                                           'timing_new2',
                                                           'age_left_school',
                                                           'age')] )
            ms_df <- mclapply(1:r, function(i)
                listFromLong(pdf_mscmt[[i]], unit.variable="pid_sampled",
                             time.variable="timing_new2",
                             unit.names.variable="pid_char") )
            times.depms  <- cbind("earnings" = c(min(u),max(u)) )
            times.predms <- cbind("earnings"          = c(min(u),max(u)),
                                  "age"           = c(min(u),max(u)),
                                  "age_left_school"   = c(min(u),max(u)))
            res <- mclapply(1:r, function(i)
                mscmt(ms_df[[i]], treatment.identifier = as.character(1),
                      controls.identifier = as.character(2:(k_controls+1)),
                      times.dep = times.depms, times.pred = times.predms, seed=1) )
            synth_obj_save[[i]] = res
            treated_w = data.frame(pidp = p_id, mscmt_w = 1)
            dff_treated_w_full = merge( isc_sample[[i]] , treated_w, by = 'pidp')
            setDT(dff_treated_w_full)
            setorder(dff_treated_w_full, pidp, timing_new, age)
            synth_w = lapply(1:r, function(i) data.frame(pid_sampled = as.numeric(names(res[[i]]$w)),
                                                         mscmt_w = res[[i]]$w, n_controls = length(unique(c_id)) ) )
            dff_controls_w_full =  lapply(1:r, function(i) merge(synth_w[[i]], g[[i]], by = c('pid_sampled')) )
            controls_avrFULL = lapply(1:r, function(i) dff_controls_w_full[[i]] %>% f_summarise_controls )
            treated_avrFULL = dff_treated_w_full %>% group_by(pidp, timing_new) %>%
                summarise(earnings = mean(earnings, na.rm = T),
                          age = mean(age, na.rm = T),
                          age_left_school = mean(age_left_school, na.rm = T),
                          age = median(age),
                          year = mean(year))
            all_synth_w_full =  lapply(1:r, function(i) merge(treated_avrFULL, controls_avrFULL[[i]], by = c('timing_new')) )
            all_synth_w_full = rbindlist(all_synth_w_full, idcol = 'boot')
            setDT(all_synth_w_full)
            all_synth_w_full[, post_treatment_period := ifelse(timing_new >= 0,1,0), pidp]
            all_synth_w_full %>% group_by(post_treatment_period) %>%
                summarise(earnings = mean(earnings), earnings_synth = mean(earnings_synth)) %>%
                mutate(diff = round(earnings - earnings_synth)) # the difference is the average individual causal effect #
            all_synth_w_full = all_synth_w_full %>% select(boot, pidp, age, timing_new, post_treatment_period, everything())
            synth_w_df[[i]] = all_synth_w_full
        }, error=function(e){cat("treated id :",i, "error")} )
    }
    return(synth_w_df)
}

r=50
max_seeds <- 1000
for(seed in 1:max_seeds) {
  synth_w_df <- run(seed, r)
  
  average <- synth_w_df[[1]] %>% group_by(pidp, timing_new) %>%
              summarise(earnings = mean(earnings), synthetic = mean(earnings_synth))
    if (seed == 1){
    df1 <- data.frame(pidp = average$pidp, timing = average$timing_new, earnings = average$earnings)
  }
  df1[ , ncol(df1) + 1] <- average$synthetic
  colnames(df1)[ncol(df1)] <- paste0("synthetic", seed)

  average <- synth_w_df[[2]] %>% group_by(pidp, timing_new) %>%
    summarise(earnings = mean(earnings), synthetic = mean(earnings_synth))
  if (seed == 1){
    df2 <- data.frame(pidp = average$pidp, timing = average$timing_new, earnings = average$earnings)
  }
  df2[ , ncol(df2) + 1] <- average$synthetic
  colnames(df2)[ncol(df2)] <- paste0("synthetic", seed)
  
  isc_df = bind_rows(synth_w_df)
  setorder(isc_df, pidp, boot, timing_new)
  average_causal <- isc_df %>%
    mutate(diff = earnings - earnings_synth) %>%
    group_by(pidp, timing_new, post_treatment_period) %>%
    summarise(m = mean(diff)) %>%
    group_by(timing_new, post_treatment_period) %>%
    summarise(round(mean(m)))  %>% as.data.frame()
  
  if (seed == 1){
    df3 <- data.frame(timing = average_causal$timing_new)
  }
  df3[ , ncol(df3) + 1] <- average_causal$`round(mean(m))`
  colnames(df3)[ncol(df3)] <- paste0("average_causal", seed)
  
  write_csv(df1, file = here("data", "vagni_breen", "output", "VB_seeds_person1.csv"))
  write_csv(df2, file = here("data", "vagni_breen", "output", "VB_seeds_person2.csv"))
  write_csv(df3, file = here("data", "vagni_breen", "output", "VB_seeds_av_causal.csv"))
}