#!/bin/bash

# Set the path to the classification directory
classification_dir="/home/mattiapggioli/lyrics2ts/data/classification"

# Iterate over all subdirectories in classification_dir
for subdir in "$classification_dir"/*/; do
    # Construct the path to the data.csv file
    data_path="$subdir/data.csv"
    echo "$data_path"
    # Check if the data.csv file exists
    if [ -f "$data_path" ]; then
        # Run the script on the data.csv file
        #python sentiment_ts.py "$data_path"
        #python features_ts.py "$data_path"
        python sbert_ts.py "$data_path" --use_sample
    fi

done