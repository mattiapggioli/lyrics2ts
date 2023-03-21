import argparse
import ast
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tslearn.utils import to_time_series_dataset, save_time_series_txt


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def extract_elements(nested_list):
    extracted_elements = []
    for sublist in nested_list:
        first_element = sublist[0]
        last_element = sublist[-1]
        middle_index = len(sublist) // 2
        middle_element = sublist[middle_index]
        remaining_elements = [
            *sublist[1:middle_index],
            *sublist[middle_index+1:-1]]
        random_element = random.choice(remaining_elements)
        extracted_elements.extend(
            [first_element, last_element, middle_element, random_element])
    return extracted_elements

def generate_ts_data(texts,pca):
    ts = []
    multi_ts_global_features = []
    for text in tqdm(texts):
        multi_ts = model.encode(text).tolist()
        t = list(map(
            lambda x: x[0], pca.transform(multi_ts)))
        ts.append(t) 
        mean_values = np.mean(multi_ts, axis=0)        
        multi_ts_global_features.append(mean_values)        
    ts = to_time_series_dataset(ts)
    multi_ts_global_features = np.array(multi_ts_global_features)
    return (multi_ts_global_features, ts)

def write_files(directory_path, ts_global_features, ts, y, pca):
    sub_dir_path = f'{directory_path}/sbert' 
    os.makedirs(sub_dir_path, exist_ok=True)    
    with open(f'{sub_dir_path}/X.npy', 'wb') as f:
        np.save(f, ts_global_features, allow_pickle=True)    
    save_time_series_txt(f'{sub_dir_path}/ts.txt', ts)
    with open(f'{sub_dir_path}/y_class.npy', 'wb') as f:
        np.save(f, y, allow_pickle=True)
    joblib.dump(pca, f'{sub_dir_path}/pca_model.pkl')
    
def write_execution_time(start_time, end_time, n_texts, directory_path):
    execution_time = end_time - start_time
    with open('execution_time.txt', 'a') as f:
        f.write('Script: {}\n'.format(__file__))
        f.write('Texts processed: {}\n'.format(n_texts))
        f.write('Execution time: {:.2f} seconds, {:.2f} minutes, {:.2f} hours\n'
            .format(execution_time, execution_time/60, execution_time/3600)) 
        f.write('Directory path: {}\n'.format(directory_path))
        f.write('-' * 50 + '\n')

def main(file_path, use_sample):
    start_time = time.time()
    df = pd.read_csv(file_path)
    directory_path = os.path.dirname(file_path)
    texts = df['text'].apply(ast.literal_eval)    
    y = np.array(df['class'])    
    txt_train = train_test_split(
        texts, y, test_size=0.20, random_state=42, stratify=y)[0].tolist()
    if use_sample:        
        txt_train = extract_elements(txt_train)
    sentence_embeddings = model.encode(txt_train)    
    pca = PCA(n_components=1)
    pca_embeddings = pca.fit(sentence_embeddings) 
    del sentence_embeddings
    multi_ts_global_features, ts = generate_ts_data(texts, pca)    
    end_time = time.time()        
    write_files(directory_path, multi_ts_global_features, ts, y, pca)
    write_execution_time(start_time, end_time, len(texts), directory_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process text data.')
    parser.add_argument('file_path', type=str, help='path to the input CSV file') 
    parser.add_argument('--use_sample', action='store_true', help='use a sample from training data')
    args = parser.parse_args()    
    main(args.file_path, args.use_sample)
