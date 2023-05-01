##########################################################################################
# Specification Curve Analysis and Digital Technology Use
# R-Script 2.3: MCS Make Specification Curves
##########################################################################################

library(tidyr)
library(dplyr)
library("heplots")
library(foreach)
library(doParallel)

resultsframe <- function(x_var, y_var) {
  # takes character vector of outcome variables as an input
  # returns the results vector of all specifications
  levels_x <- x_var
  levels_y <- y_var
  levels_c <- c("No Controls", "Controls")
  combinations <-
    length(levels_x) * length(levels_y) * length(levels_c)
  results_frame <- data.frame(matrix(NA, nrow = combinations, ncol = 9))
  colnames(results_frame) <- c("controls", "effect", "y_variable", "x_variable")
  results_frame$x_variable <- rep(levels_x,
                                  each = nrow(results_frame) / length(levels_x))
  results_frame$y_variable <- rep(rep(levels_y,
                                      each = nrow(results_frame) / (length(levels_x) * length(levels_y))),
                                  times = length(levels_x))
  results_frame$controls <- rep(rep(rep(levels_c,
                                        each = nrow(results_frame) / (length(levels_x) * length(levels_y) * length(levels_c))),
                                    times = length(levels_x)), times = length(levels_y))
  return(results_frame)
}

curve <- function(input, data_short) {
  # takes results frame as an input
  # returns results frame including the specification curve analysis results
  results_frame <- input
  for (n in 1:nrow(results_frame)) {
#    print(n/nrow(results_frame))
    data_short$dv <-
      rowMeans(subset(data_short, select = results_frame$y_variable[[n]]),
               na.rm = FALSE)
    data_short$iv <-
      subset(data_short, select = results_frame$x_variable[[n]])
    if (results_frame$controls[n] == "No Controls") {
      reg <- lm(scale(dv) ~ scale(iv), data = data_short)
    } else if (results_frame$controls[n] == "Controls") {
      reg <- lm(
        scale(dv) ~ scale(iv) + scale(edumot) +
          scale(fd06e00) + scale(clpar) + scale(fcpaab00) +
          scale(fpwrdscm) + scale(fdacaq00) + scale(fd05s00) +
          scale(fpclsi00) + scale(fpchti00) + scale(fdkessl) + scale(fdtots00) +
          scale(foede000),
        data = data_short
      )
    }
    results_frame$effect[n] <- summary(reg)$coef[[2, 1]] %>% {ifelse(. == 0, NA, .)}
  }
  return(results_frame)
}

main_fun <- function(data, i) {
  print(i)
  set.seed(i)
  x_variables <- c("fctvho00r", "fccomh00r", "fccmex00r", "fcinth00r", "fcsome00r", "tech")
  x_names <- c("Weekday TV", "Weekday Electronic Games", "Own Computer",
               "Use Internet at Home", "Hours of Social Media Use", "tech")
  y <- c("fcmdsa00r", "fcmdsb00r", "fcmdsc00r", "fcmdsd00r", "fcmdse00r",
         "fcmdsf00r", "fcmdsg00r", "fcmdsh00r", "fcmdsi00r", "fcmdsj00r",
         "fcmdsk00r", "fcmdsl00r", "fcmdsm00r", "fcsati00r", "fcgdql00r",
         "fcdowl00r", "fcvalu00r", "fcgdsf00r", "fcscwk00r", "fcwylk00r",
         "fcfmly00r", "fcfrns00r", "fcschl00r", "fclife00r")
  y_variables <- (do.call("c", lapply(seq_along(y), function(i)
    combn(y, i, FUN = list))))

  y_variables_sample <-
    sample(y_variables[-(1:length(y))], 806, replace = FALSE)
  y4 <- y_variables[1:length(y)]
  y1 <- c( "fcmdsa00r", "fcmdsb00r", "fcmdsc00r", "fcmdsd00r", "fcmdse00r",
           "fcmdsf00r", "fcmdsg00r", "fcmdsh00r", "fcmdsi00r", "fcmdsj00r",
           "fcmdsk00r", "fcmdsl00r", "fcmdsm00r")
  y2 <- c("fcsati00r", "fcgdql00r", "fcdowl00r", "fcvalu00r", "fcgdsf00r")
  y3 <- c("fcscwk00r", "fcwylk00r", "fcfmly00r", "fcfrns00r", "fcschl00r", "fclife00r")
  y_variables_sample_cm <- c(y_variables_sample, y4, list(y1, y2, y3))
  rm(y_variables, y1, y2, y3, y4)
  controls <- c("edumot", "fd06e00", "clpar", "fcpaab00", "fpwrdscm", "fdacaq00",
                "fd05s00", "fpclsi00", "fpchti00", "fdkessl", "fdtots00", "foede000")

  data_short <- data[, c( "fctvho00r", "fccomh00r", "fccmex00r", "fcinth00r",
                          "fcsome00r","fcmdsa00r", "fcmdsb00r", "fcmdsc00r",
                          "fcmdsd00r", "fcmdse00r", "fcmdsf00r", "fcmdsg00r",
                          "fcmdsh00r", "fcmdsi00r", "fcmdsj00r", "fcmdsk00r",
                          "fcmdsl00r", "fcmdsm00r", "fcsati00r", "fcgdql00r",
                          "fcdowl00r", "fcvalu00r", "fcgdsf00r", "fcscwk00r",
                          "fcwylk00r", "fcfmly00r", "fcfrns00r", "fcschl00r",
                          "fclife00r", "edumot", "fd06e00", "clpar", "fcpaab00",
                          "fpwrdscm", "fdacaq00","fd05s00","fpwrdscm",
                          "fpclsi00", "fpchti00", "fdkessl","fdtots00",
                          "foede000", "tech")]
  results_mcs_sca_cm <- curve(resultsframe(x_var = x_variables,
                                           y_var = y_variables_sample_cm),
                              data_short)

  y <- c("fpsdpf00", "fpsdro00", "fpsdhs00", "fpsdsr00","fpsdtt00",
         "fpsdsp00", "fpsdor00", "fpsdmw00", "fpsdhu00", "fpsdfs00",
         "fpsdgf00", "fpsdfb00", "fpsdud00", "fpsdlc00", "fpsddc00",
         "fpsdnc00", "fpsdky00", "fpsdoa00", "fpsdpb00", "fpsdvh00",
         "fpsdst00", "fpsdcs00", "fpsdgb00", "fpsdfe00","fpsdte00")
  y_variables <- (do.call("c", lapply(seq_along(y), function(i) combn(y, i, FUN = list))))
  y_variables_sample <- sample(y_variables[-(1:length(y))], 801, replace = FALSE)
  y1 <- y_variables[1:length(y)]
  y2 <- c("fconduct")
  y3 <- c("fhyper")
  y4 <- c("fpeer")
  y5 <- c("fprosoc")
  y6 <- c("febdtot") # total
  y7 <- c("femotion")
  y8 <- c("femotion", "fpeer")
  y9 <- c("fconduct", "fhyper")
  y_variables_sample_pr <- c(y_variables_sample, y1,
                             list(y2,y3,y4,y5,y6,y7,y8,y9))
  data_short <- data[, c( "fctvho00r", "fccomh00r", "fccmex00r", "fcinth00r",
                          "fcsome00r", "tech", "fpsdpf00", "fpsdro00",
                          "fpsdhs00", "fpsdsr00", "fpsdtt00", "fpsdsp00",
                          "fpsdor00", "fpsdmw00", "fpsdhu00", "fpsdfs00",
                          "fpsdgf00", "fpsdfb00", "fpsdud00", "fpsdlc00",
                          "fpsddc00", "fpsdnc00", "fpsdky00", "fpsdoa00",
                          "fpsdpb00", "fpsdvh00", "fpsdst00", "fpsdcs00",
                          "fpsdgb00", "fpsdfe00", "fpsdte00", "fconduct",
                          "fhyper",    "fpeer", "fprosoc", "febdtot",
                          "femotion", "edumot", "fd06e00", "clpar", "fcpaab00",
                          "fpwrdscm", "fdacaq00", "fd05s00", "fpwrdscm",
                          "fpclsi00", "fpchti00", "fdkessl", "fdtots00",
                          "foede000")]
  results_mcs_sca_pr <-  curve(resultsframe(x_var = x_variables,
                                            y_var = y_variables_sample_pr),
                               data_short)
  output.dir <- file.path(getwd(),'..',  'data', 'mcs', 'results')
  save(results_mcs_sca_cm,
       file = file.path(output.dir, sprintf("2_3_sca_mcs_results_cm_seed_%s.rda", i)))
  save(results_mcs_sca_pr,
       file = file.path(output.dir, sprintf("2_3_sca_mcs_results_pr_seed_%s.rda", i)))
}

private.data.dir <- file.path(getwd(), '..', 'data', 'mcs', 'raw')
data <- read.csv(file.path(private.data.dir, "1_3_prep_mcs_data.csv"), header=TRUE, sep = ",")

#n.cores=detectCores() - 2
#my.cluster <- parallel::makeCluster(
#  n.cores,
#  type = "PSOCK"
#)
#registerDoParallel(my.cluster)
#foreach(i=1:10000) %dopar% {
#  main_fun(data, i)
#}
#foreach(i=1:10000) %dopar% {
#  main_fun(data, i)
#}
#stopCluster(my.cluster)


for(i in 8630:10000){
  main_fun(data, i)
}
