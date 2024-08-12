library(dqrng)

seeds <- as.integer(readLines(file.path("..", "assets", "seed_list.txt")))

# Set the number of iterations
num_iterations <- 10000 # length(seeds)
n <- 1000000

# Initialize lists and vectors to store the output
output_list_32 <- vector("list", length = num_iterations)

# Record the start time
start_time <- Sys.time()

# Create a text progress bar
pb <- txtProgressBar(min = 0, max = num_iterations, style = 3)

# Loop through the specified number of iterations
for (i in 1:num_iterations) {
  
  # Set a new seed for each iteration
  set.seed(seeds[i])

  # Generate random numbers
  U_32 <- runif(n)

  # Identify duplicates
  I_32 <- duplicated(U_32)
  
  # Store the cumulative sums in the list
  output_list_32[[i]] <- cumsum(duplicated(U_32))

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

# Save the final row of each dataset to CSV
# Extract final row for 32-bit results
final_row_32 <- sapply(output_list_32, function(x) x[n])

# Define output file paths for final rows
output_file_final_row_32 <- file.path("..", "data", "collisions", "stats_32_final_row.csv")

# Save the final rows to CSV files
write.csv(final_row_32, file = output_file_final_row_32, row.names = FALSE)

# Print confirmation message
cat("Final rows have been saved to CSV files.\n")

# Convert list to matrix for row-wise statistics
output_matrix_32 <- do.call(cbind, output_list_32)

# Compute row-wise statistics for 32-bit data
stats_32 <- data.frame(
  min = apply(output_matrix_32, 1, min),
  max = apply(output_matrix_32, 1, max),
  median = apply(output_matrix_32, 1, median),
  mean = apply(output_matrix_32, 1, mean)
)

# Define output file paths for statistics
output_file_stats_32 <- file.path("..", "data", "collisions", "stats_32bit_rowwise.csv")

# Save the statistics to CSV files
write.csv(stats_32, file = output_file_stats_32, row.names = FALSE)

# Print confirmation message
cat("Row-wise statistics have been saved to CSV files.\n")
