library(tidyverse)
library(haven)
library(readstata13)

# ---------------------------
# Load data (source: https://github.com/Jiahui1902/hte_machinelearning)
# ---------------------------

edurose_mediation_20181126 <- read.dta13(
  "../data/sample_edurose_mediation_20181126.dta",
  convert.factors = FALSE
) %>% 
  as.data.frame()

linear_terms <- c("male","black","hisp","i_daded",
                  "i_momed","i_parinc","i_daduwhcol",
                  "i_intact","i_sibsz","i_rural",
                  "i_south","i_abil","i_hsprog",
                  "i_eduexp","i_eduasp","i_freduasp",
                  "i_rotter","i_delinq","i_schdisadv",
                  "i_mar18","i_parent18","good")

outcomevariable <- "lowwaprop"
indicatorvariable <- "compcoll25"
ps_indicator <- "propsc_com25"
covariates <- c(linear_terms, ps_indicator)

data <- edurose_mediation_20181126[
  , c(outcomevariable, covariates, indicatorvariable)
]

# ---------------------------
# Insert missingness only into covariates
# ---------------------------

covariates <- c(linear_terms, ps_indicator)

# number of cells in covariates only
n_cov_cells <- nrow(data) * length(covariates)

# choose proportion of missingness
prop_miss <- 0.10
n_na <- ceiling(n_cov_cells * prop_miss)

# sample positions in the covariate block
cov_positions <- arrayInd(
  sample(n_cov_cells, n_na),
  .dim = c(nrow(data), length(covariates))
)

data_miss <- data

# convert covariate positions to actual column indices in data_miss
cov_col_idx <- match(covariates, names(data_miss))

# build matrix of (row, col) indices
idx <- cbind(
  cov_positions[, 1],
  cov_col_idx[cov_positions[, 2]]
)

# elementwise replacement using matrix indexing
data_miss[idx] <- NA

# extra 15% missingness in i_parinc
set.seed(123)  # optional
data_miss$i_parinc[
  sample(nrow(data), floor(nrow(data) * 0.15))
] <- NA

# ---------------------------
# Write files
# ---------------------------
write.csv(data,
          "../data/brand_et_al_sample.csv", row.names = FALSE)
write.csv(data_miss,
          "../data//brand_et_al_with_missing_sample.csv", row.names = FALSE)