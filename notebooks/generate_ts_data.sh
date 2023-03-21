#!/bin/bash

# Set the path to the classification directory
classification_dir="/home/mattiapggioli/lyrics2ts/data/classification"

# First iteration: run the first script for all data.csv files
for subdir in "$classification_dir"/*/; do
    # Construct the path to the data.csv file
    data_path="$subdir/data.csv"
    echo "$data_path"
    # Check if the data.csv file exists
    if [ -f "$data_path" ]; then
        # Run the first script on the data.csv file
        python sentiment_ts.py "$data_path"
    fi
done

# Second iteration: run the second script for all data.csv files
for subdir in "$classification_dir"/*/; do
    # Construct the path to the data.csv file
    data_path="$subdir/data.csv"
    echo "$data_path"
    # Check if the data.csv file exists
    if [ -f "$data_path" ]; then
        # Run the second script on the data.csv file
        python features_ts.py "$data_path"
    fi
done

# Third iteration: run the third script for all data.csv files
for subdir in "$classification_dir"/*/; do
    # Construct the path to the data.csv file
    data_path="$subdir/data.csv"
    echo "$data_path"
    # Check if the data.csv file exists
    if [ -f "$data_path" ]; then
        # Run the third script on the data.csv file
        python sbert_ts.py "$data_path" --use_sample
    fi
done
