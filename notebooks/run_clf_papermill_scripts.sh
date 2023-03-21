#!/bin/bash

# Loop through all directories inside "classification"
for dir in /home/mattiapggioli/lyrics2ts/data/classification/*/; do
  # Get the directory name
  dir_name=$(basename "$dir")
  
  # Loop through all subdirectories named "sbert", "sentiment", or "features"
  for sub_dir in "$dir"/{sbert,sentiment,features}/; do
    # Get the subdirectory name
    sub_dir_name=$(basename "$sub_dir")
    
    # Check if this is one of the specified directories
    if [[ "$dir_name" == "pp-rc" || "$dir_name" == "pp-rc-rp" || "$dir_name" == "rc-rp" || "$dir_name" == "rock-subgenres" ]]; then
      # Construct the papermill command with the additional parameters
      output_file="../clf-reports/${dir_name}_${sub_dir_name}.ipynb"
      data_path=$(echo "$sub_dir" | sed 's|/\{1,\}$||; s|//|/|g; s|$|/|')
      command="papermill 09_ts_classification.ipynb \"$output_file\" -p data_path \"$data_path\" -p paa_window_size 20 -p undersampling 0.3"
    else
      # Construct the papermill command without the additional parameters
      output_file="../clf-reports/${dir_name}_${sub_dir_name}.ipynb"
      data_path=$(echo "$sub_dir" | sed 's|/\{1,\}$||; s|//|/|g; s|$|/|')
      command="papermill 09_ts_classification.ipynb \"$output_file\" -p data_path \"$data_path\""
    fi
    
    # Print the papermill command
    echo "Running command:"
    echo "$command"
    
    # Run the papermill command
    eval "$command"
  done
done
