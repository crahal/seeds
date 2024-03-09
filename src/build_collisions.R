# Based on: https://arxiv.org/pdf/2003.08009.pdf
library(dqrng)

seeds <- as.integer(readLines(file.path("..", "assets", "seed_list.txt")))

# Set the number of iterations
num_iterations <- 100 # length(seeds)
n <- 1000000

# Initialize a list to store the output (cumulative sums of duplicates)
output_list_32 <- vector("list", length = num_iterations)
output_list_64 <- numeric(num_iterations)

# Record the start time
start_time <- Sys.time()

# Create a text progress bar
pb <- txtProgressBar(min = 0, max = num_iterations, style = 3)

# Loop through the specified number of iterations

for (i in 1:num_iterations) {
  
  # Set a new seed for each iteration
  set.seed(seeds[i])
  dqset.seed(seeds[i])
  
  # Generate random numbers
  U_32 <- runif(n)
  U_64 <- dqrunif(n)
  
  # Identify duplicates
  I_32 <- duplicated(U_32)
  
  # Store the cumulative sums in the list
  output_list_32[[i]] <- cumsum(duplicated(U_32))
  output_list_64[i] <- sum(duplicated(U_64))
  # Increment the progress bar
  setTxtProgressBar(pb, i)
  
  # Calculate time elapsed
  time_elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  # Calculate time remaining (estimated)
  time_remaining <- (time_elapsed / i) * (num_iterations - i)
  
  # Print time information, overwriting the previous output
  cat(sprintf("\rTime Elapsed: %.2f seconds | Time Remaining: %.2f seconds", time_elapsed, time_remaining))
  flush.console()  # Ensure the output is immediately printed
  
}

# Close the progress bar
close(pb)

# Convert to data frame with sequential column names
#output_df_32 <- as.data.frame(output_list_32)
#colnames(output_df_32) <- 1:ncol(output_df_32)

# Save the data frame to a CSV file
output_file_32 <- file.path("..", "data", "collisions", "output_list_32_R.csv")
write.csv(output_list_32, file = output_file_32, row.names = FALSE)

output_file_64 <- file.path("..", "data", "collisions", "output_array_64_R.csv")
write.csv(output_list_64, file = output_file_64, row.names = FALSE)
