# Specification Curve Analysis and Digital Technology Use
# R-Script 1.3 (adapted): MCS clean data and make variables
#
# Paths adapted to this repo:
#   - input .tab files expected under ../data/mcs/raw/
#   - output CSV written to ../data/mcs/1_3_prep_mcs_data.csv
# Notes:
#   - This assumes the raw MCS wave 6 files are saved as .tab (TSV) with the
#     same basenames as the original CSVs (e.g., mcs6_cm_assessment.tab).
#   - Requires R packages: tidyverse, psych, foreign (foreign is loaded but
#     not used in this script).

library("foreign")
library("tidyverse")
library("psych")

# Set base paths
folder_data <- file.path("..", "data", "mcs", "raw")
out_path <- file.path(folder_data, "1_3_prep_mcs_data.csv")

# Helper to read .tab (TSV) files
read_tab <- function(name) {
  read.table(file.path(folder_data, paste0(name, ".tab")),
             header = TRUE, sep = "\t", stringsAsFactors = FALSE)
}

#######################################################
# Read MCS Dataset (.tab instead of .csv)
#######################################################
data1  <- read_tab("mcs6_cm_assessment")
data2  <- read_tab("mcs6_cm_derived")
data3  <- read_tab("mcs6_cm_interview")
data4  <- read_tab("mcs6_cm_measurement")
data5  <- read_tab("mcs6_family_derived")
data6  <- read_tab("mcs6_parent_cm_interview")
data7  <- read_tab("mcs6_parent_assessment")
data8  <- read_tab("mcs6_parent_derived")
data9  <- read_tab("mcs6_parent_interview")
data10 <- read_tab("mcs6_proxy_partner_interview")
data11 <- read_tab("mcs6_hhgrid")

# change names of columns to lowercase
to_lower_names <- function(df) {
  names(df) <- tolower(names(df))
  df
}
data1  <- to_lower_names(data1)
data2  <- to_lower_names(data2)
data3  <- to_lower_names(data3)
data4  <- to_lower_names(data4)
data5  <- to_lower_names(data5)
data6  <- to_lower_names(data6)
data7  <- to_lower_names(data7)
data8  <- to_lower_names(data8)
data9  <- to_lower_names(data9)
data10 <- to_lower_names(data10)
data11 <- to_lower_names(data11)

#######################################################
# Combine different datasets
#######################################################
# Cohort Members
data1$mcsid1 <- ifelse(
  data1$fcnum00 == 1, paste(as.character(data1$mcsid), "_1", sep = ""),
  ifelse(
    data1$fcnum00 == 2, paste(as.character(data1$mcsid), "_2", sep = ""),
    ifelse(data1$fcnum00 == 3, paste(as.character(data1$mcsid), "_3", sep = ""), NA)
  )
)
data1$mcsid <- as.character(data1$mcsid)
data2$mcsid2 <- ifelse(
  data2$fcnum00 == 1, paste(as.character(data2$mcsid), "_1", sep = ""),
  ifelse(
    data2$fcnum00 == 2, paste(as.character(data2$mcsid), "_2", sep = ""),
    ifelse(data2$fcnum00 == 3, paste(as.character(data2$mcsid), "_3", sep = ""), NA)
  )
)
data3$mcsid3 <- ifelse(
  data3$fcnum00 == 1, paste(as.character(data3$mcsid), "_1", sep = ""),
  ifelse(
    data3$fcnum00 == 2, paste(as.character(data3$mcsid), "_2", sep = ""),
    ifelse(data3$fcnum00 == 3, paste(as.character(data3$mcsid), "_3", sep = ""), NA)
  )
)
data4$mcsid4 <- ifelse(
  data4$fcnum00 == 1, paste(as.character(data4$mcsid), "_1", sep = ""),
  ifelse(
    data4$fcnum00 == 2, paste(as.character(data4$mcsid), "_2", sep = ""),
    ifelse(data4$fcnum00 == 3, paste(as.character(data4$mcsid), "_3", sep = ""), NA)
  )
)
data5$mcsid5 <- as.character(data5$mcsid)

# Parents
data6$mcsid6  <- paste(as.character(data6$mcsid), as.character(data6$fpnum00), sep = "_")
data7$mcsid7  <- paste(as.character(data7$mcsid), as.character(data7$fpnum00), sep = "_")
data8$mcsid8  <- paste(as.character(data8$mcsid), as.character(data8$fpnum00), sep = "_")
data9$mcsid9  <- paste(as.character(data9$mcsid), as.character(data9$fpnum00), sep = "_")
data10$mcsid10 <- paste(as.character(data10$mcsid), as.character(data10$fpnum00), sep = "_")

# Merge Cohort Members
data_cm <- dplyr::left_join(data1,  data2[, is.na(match(names(data2),  names(data1)))],  by = c("mcsid1" = "mcsid2"))
data_cm <- dplyr::left_join(data_cm, data3[, is.na(match(names(data3), names(data_cm)))], by = c("mcsid1" = "mcsid3"))
data_cm <- dplyr::left_join(data_cm, data4[, is.na(match(names(data4), names(data_cm)))], by = c("mcsid1" = "mcsid4"))
data_cm <- dplyr::left_join(data_cm, data5[, is.na(match(names(data5), names(data_cm)))], by = c("mcsid"  = "mcsid5"))

# Merge Parents
data_pa <- dplyr::left_join(data6, data7[, is.na(match(names(data7), names(data6)))], by = c("mcsid6" = "mcsid7"))
data_pa <- dplyr::left_join(data_pa, data8[, is.na(match(names(data8), names(data_pa)))], by = c("mcsid6" = "mcsid8"))
data_pa <- dplyr::left_join(data_pa, data9[, is.na(match(names(data9), names(data_pa)))], by = c("mcsid6" = "mcsid9"))

data_pa$mcsid1_r <- ifelse(
  data_pa$fcnum00 == 1, paste(as.character(data_pa$mcsid), "_1", sep = ""),
  ifelse(
    data_pa$fcnum00 == 2, paste(as.character(data_pa$mcsid), "_2", sep = ""),
    ifelse(data_pa$fcnum00 == 3, paste(as.character(data_pa$mcsid), "_3", sep = ""), NA)
  )
)
data_pa$fpnum00_r <- ifelse(data_pa$fpnum00 == 1, 1, 0)
data_pa_1 <- dplyr::filter(data_pa, fpnum00_r == 1)

# Merge cohort members and parents, not merging duplicate rows
data <- dplyr::left_join(
  data_cm,
  data_pa_1[, is.na(match(names(data_pa_1), names(data_cm)))],
  by = c("mcsid1" = "mcsid1_r")
)

rm(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11)
gc()

#######################################################
# Set missing values: any negative numbers
#######################################################
is.na(data[, ]) <- data[, ] < 0

#######################################################
# Recode Well-being Measures: Cohort Member
#######################################################
# Scale to 10-point and reverse
data$fcscwk00r <- (10 - 1) * (data$fcscwk00 - 1) / (7 - 1) + 1
data$fcwylk00r <- (10 - 1) * (data$fcwylk00 - 1) / (7 - 1) + 1
data$fcfmly00r <- (10 - 1) * (data$fcfmly00 - 1) / (7 - 1) + 1
data$fcfrns00r <- (10 - 1) * (data$fcfrns00 - 1) / (7 - 1) + 1
data$fcschl00r <- (10 - 1) * (data$fcschl00 - 1) / (7 - 1) + 1
data$fclife00r <- (10 - 1) * (data$fclife00 - 1) / (7 - 1) + 1

data$fcsati00r <- (10 - 1) * (data$fcsati00 - 1) / (4 - 1) + 1
data$fcgdql00r <- (10 - 1) * (data$fcgdql00 - 1) / (4 - 1) + 1
data$fcdowl00r <- (10 - 1) * (data$fcdowl00 - 1) / (4 - 1) + 1
data$fcvalu00r <- (10 - 1) * (data$fcvalu00 - 1) / (4 - 1) + 1
data$fcgdsf00r <- (10 - 1) * (data$fcgdsf00 - 1) / (4 - 1) + 1

data$fcmdsa00r <- (10 - 1) * (data$fcmdsa00 - 1) / (3 - 1) + 1
data$fcmdsb00r <- (10 - 1) * (data$fcmdsb00 - 1) / (3 - 1) + 1
data$fcmdsc00r <- (10 - 1) * (data$fcmdsc00 - 1) / (3 - 1) + 1
data$fcmdsd00r <- (10 - 1) * (data$fcmdsd00 - 1) / (3 - 1) + 1
data$fcmdse00r <- (10 - 1) * (data$fcmdse00 - 1) / (3 - 1) + 1
data$fcmdsf00r <- (10 - 1) * (data$fcmdsf00 - 1) / (3 - 1) + 1
data$fcmdsg00r <- (10 - 1) * (data$fcmdsg00 - 1) / (3 - 1) + 1
data$fcmdsh00r <- (10 - 1) * (data$fcmdsh00 - 1) / (3 - 1) + 1
data$fcmdsi00r <- (10 - 1) * (data$fcmdsi00 - 1) / (3 - 1) + 1
data$fcmdsj00r <- (10 - 1) * (data$fcmdsj00 - 1) / (3 - 1) + 1
data$fcmdsk00r <- (10 - 1) * (data$fcmdsk00 - 1) / (3 - 1) + 1
data$fcmdsl00r <- (10 - 1) * (data$fcmdsl00 - 1) / (3 - 1) + 1
data$fcmdsm00r <- (10 - 1) * (data$fcmdsm00 - 1) / (3 - 1) + 1

data$fcscwk00r <- 11 - data$fcscwk00r
data$fcwylk00r <- 11 - data$fcwylk00r
data$fcfmly00r <- 11 - data$fcfmly00r
data$fcfrns00r <- 11 - data$fcfrns00r
data$fcschl00r <- 11 - data$fcschl00r
data$fclife00r <- 11 - data$fclife00r

data$fcsati00r <- 11 - data$fcsati00r
data$fcgdql00r <- 11 - data$fcgdql00r
data$fcdowl00r <- 11 - data$fcdowl00r
data$fcvalu00r <- 11 - data$fcvalu00r
data$fcgdsf00r <- 11 - data$fcgdsf00r

data$fcmdsa00r <- 11 - data$fcmdsa00r
data$fcmdsb00r <- 11 - data$fcmdsb00r
data$fcmdsc00r <- 11 - data$fcmdsc00r
data$fcmdsd00r <- 11 - data$fcmdsd00r
data$fcmdse00r <- 11 - data$fcmdse00r
data$fcmdsf00r <- 11 - data$fcmdsf00r
data$fcmdsg00r <- 11 - data$fcmdsg00r
data$fcmdsh00r <- 11 - data$fcmdsh00r
data$fcmdsi00r <- 11 - data$fcmdsi00r
data$fcmdsj00r <- 11 - data$fcmdsj00r
data$fcmdsk00r <- 11 - data$fcmdsk00r
data$fcmdsl00r <- 11 - data$fcmdsl00r
data$fcmdsm00r <- 11 - data$fcmdsm00r

#######################################################
# Recode Well-being measures: Parent
#######################################################
data$fpsdro00 <- 2 - data$fpsdro00
data$fpsdhs00 <- 2 - data$fpsdhs00
data$fpsdtt00 <- 2 - data$fpsdtt00
data$fpsdsp00 <- 2 - data$fpsdsp00
data$fpsdmw00 <- 2 - data$fpsdmw00
data$fpsdfs00 <- 2 - data$fpsdfs00
data$fpsdfb00 <- 2 - data$fpsdfb00
data$fpsdud00 <- 2 - data$fpsdud00
data$fpsddc00 <- 2 - data$fpsddc00
data$fpsdnc00 <- 2 - data$fpsdnc00
data$fpsdoa00 <- 2 - data$fpsdoa00
data$fpsdpb00 <- 2 - data$fpsdpb00
data$fpsdcs00 <- 2 - data$fpsdcs00
data$fpsdgb00 <- 2 - data$fpsdgb00
data$fpsdfe00 <- 2 - data$fpsdfe00

data$fconduct <- 11 - data$fconduct
data$fhyper   <- 11 - data$fhyper
data$fpeer    <- 11 - data$fpeer
data$femotion <- 11 - data$femotion
data$febdtot  <- 41 - data$febdtot

#######################################################
# Recode Digital Technology Use measures
#######################################################
data$fccmex00r <- 3 - data$fccmex00

data$fctvho00r <- (10 - 1) * (data$fctvho00 - 1) / (8 - 1) + 1
data$fccomh00r <- (10 - 1) * (data$fccomh00 - 1) / (8 - 1) + 1
data$fcinth00r <- (10 - 1) * (data$fcinth00 - 1) / (8 - 1) + 1
data$fcsome00r <- (10 - 1) * (data$fcsome00 - 1) / (8 - 1) + 1
data$fccmex00r <- (10 - 1) * (data$fccmex00r - 1) / (2 - 1) + 1

data$tech <- rowMeans(subset(
  data,
  select = c("fctvho00r", "fccomh00r", "fccmex00r", "fcinth00r", "fcsome00r")
), na.rm = FALSE)

#######################################################
# Recode Control measures
#######################################################
data$fd06e00 <- ifelse(data$fd06e00 == 1, 1, 0)

data$fcscbe00r <- 6 - data$fcscbe00
data$fcsint00r <- 6 - data$fcsint00
data$edumot <- rowMeans(subset(
  data,
  select = c("fcscbe00r", "fcsint00r", "fcsunh00", "fcstir00", "fcscwa00", "fcmnwo00")
), na.rm = FALSE)

is.na(data[, c("fcrlqm00", "fcrlqf00")]) <- data[, c("fcrlqm00", "fcrlqf00")] == 5
data$clpar <- rowMeans(subset(data, select = c("fcrlqm00", "fcrlqf00", "fcquam00", "fcquaf00")), na.rm = FALSE)

data$fcsltr00r <- 7 - data$fcsltr00
data$sldif <- rowMeans(subset(data, select = c("fcslln00", "fcsltr00r")), na.rm = FALSE)

data$fccsex00r <- 2 - data$fccsex00

#######################################################
# Recode Comparison Variables
#######################################################
data$fcslwk00r <- ifelse(
  data$fcslwk00 == 1, 12 - 8.5,
  ifelse(
    data$fcslwk00 == 2, 12 - 9.5,
    ifelse(
      data$fcslwk00 == 3, 12 - 10.5,
      ifelse(data$fcslwk00 == 4, 12 - 11.5, 12 - 12.5)
    )
  )
)

data$fcwuwk00r <- ifelse(
  data$fcwuwk00 == 1, 5.5,
  ifelse(data$fcwuwk00 == 2, 6.5,
         ifelse(data$fcwuwk00 == 3, 7.5,
                ifelse(data$fcwuwk00 == 4, 8.5, 9.5)))
)
data$sleeptime <- data$fcslwk00r + data$fcwuwk00r

data$hand <- ifelse(data$fchand00 == 1, 0, ifelse(data$fchand00 == 3, 1, 2))

data$fccycf00r <- 9 - data$fccycf00
data$fcglas00r <- 2 - data$fcglas00
data$fcares00r <- 2 - data$fcares00
data$fccybu00r <- 7 - data$fccybu00
data$fchurt00r <- 7 - data$fchurt00
data$fccanb00r <- 2 - data$fccanb00
data$fcalfv00r <- 2 - data$fcalfv00

#######################################################
# Save as CSV
#######################################################
dir.create(folder_data, recursive = TRUE, showWarnings = FALSE)
write.csv(file = out_path, data)
