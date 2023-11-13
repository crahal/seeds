seeds <- as.integer(readLines(file.path("..", "assets", "seed_list.txt")))

# Set the number of iterations
num_iterations <- 1000 # length(seeds)
n <- 1000000

# Initialize a list to store the output (cumulative sums of duplicates)
output_list <- vector("list", length = num_iterations)

# Record the start time
start_time <- Sys.time()

# Create a text progress bar
pb <- txtProgressBar(min = 0, max = num_iterations, style = 3)

# Loop through the specified number of iterations
for (i in 1:num_iterations) {
  # Set a new seed for each iteration
  set.seed(seeds[i])
  
  # Generate random numbers
  U <- runif(n)
  
  # Identify duplicates
  I <- duplicated(U)
  
  # Store the cumulative sums in the list
  output_list[[i]] <- cumsum(I)
  
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

cat("\n")  # Move to the next line after the loop

# Save the list of cumulative sums to a CSV file
max_length <- max(sapply(output_list, length))
output_list_padded <- lapply(output_list, function(x) c(x, rep(NA, max_length - length(x))))

# Convert to data frame with sequential column names
output_df <- as.data.frame(output_list_padded)
colnames(output_df) <- 1:ncol(output_df)

# Save the data frame to a CSV file
output_file <- file.path("..", "data", "collisions", "output_list_R.csv")
write.csv(output_df, file = output_file, row.names = FALSE)
