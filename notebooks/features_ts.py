import argparse
import ast
import os
import time
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm
from lyrics_stats import lyrics_statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tslearn.utils import to_time_series_dataset, save_time_series_txt


def get_features(text):    
    features_in_text = lyrics_statistics(text)
    features_in_text = [list(item) for item in zip(*
        [features_in_text[k] for k in features_in_text.keys()])]
    return features_in_text

def generate_ts_data(texts, scaler, pca):
    ts = []
    multi_ts_global_features = []
    for text in tqdm(texts):
        multi_ts = get_features(text) 
        t = list(map(lambda x: x[0], pca.transform(
            scaler.transform(multi_ts))))
        ts.append(t)        
        mean_values = np.mean(multi_ts, axis=0)        
        multi_ts_global_features.append(mean_values)        
    ts = to_time_series_dataset(ts)
    multi_ts_global_features = np.array(multi_ts_global_features)
    return (multi_ts_global_features, ts)

def write_files(directory_path, ts_global_features, ts, y, scaler, pca):
    sub_dir_path = f'{directory_path}/features' 
    os.makedirs(sub_dir_path, exist_ok=True)    
    with open(f'{sub_dir_path}/X.npy', 'wb') as f:
        np.save(f, ts_global_features, allow_pickle=True)    
    save_time_series_txt(f'{sub_dir_path}/ts.txt', ts)
    with open(f'{sub_dir_path}/y_class.npy', 'wb') as f:
        np.save(f, y, allow_pickle=True)
    joblib.dump(pca, f'{sub_dir_path}/pca.pkl')
    joblib.dump(scaler, f'{sub_dir_path}/scaler.pkl')

def write_execution_time(start_time, end_time, n_texts, directory_path):
    execution_time = end_time - start_time
    with open('execution_time.txt', 'a') as f:
        f.write('Script: {}\n'.format(__file__))
        f.write('Texts processed: {}\n'.format(n_texts))
        f.write('Execution time: {:.2f} seconds, {:.2f} minutes, {:.2f} hours\n'
            .format(execution_time, execution_time/60, execution_time/3600)) 
        f.write('Directory path: {}\n'.format(directory_path))
        f.write('-' * 50 + '\n')

def main(file_path):
    start_time = time.time()
    df = pd.read_csv(file_path)
    directory_path = os.path.dirname(file_path)
    texts = df['text'].apply(ast.literal_eval)    
    y = np.array(df['class'])
    txt_train = train_test_split(
        texts, y, test_size=0.20, random_state=42)[0]    
    # Create flat X_train of features by timestamp 
    X_train = np.array([sentence for text in [get_features(
        text) for text in tqdm(txt_train)] for sentence in text])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_train)
    del X_train
    multi_ts_global_features, ts = generate_ts_data(texts, scaler, pca)    
    end_time = time.time() 
    write_files(directory_path, multi_ts_global_features, ts, y, scaler, pca)
    write_execution_time(start_time, end_time, len(texts), directory_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process text data.')
    parser.add_argument('file_path', type=str, help='path to the input CSV file')
    args = parser.parse_args()
    main(args.file_path)