# set_wd to ./seeds/data/results/mcs first!

filenames = dir(pattern="*.rda")
for (i in filenames){
  dat <- load(i)
  write.csv(results_mcs_sca_cm$effect, paste('csvs/', i, '.csv', sep=""))
}
