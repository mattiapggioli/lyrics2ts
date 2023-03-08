#!/bin/bash
papermill 00_data_preparation.ipynb 00_data_preparation.ipynb
papermill 01_evaluation_ds_preparation.ipynb 01_evaluation_ds_preparation.ipynb
papermill 02_doc2vec_training.ipynb 02_doc2vec_training.ipynb 
papermill 03_gpca_model_training.ipynb 03_gpca_model_training.ipynb
papermill 04_features_ts_generation.ipynb 04_features_ts_generation.ipynb
papermill 05_sentence_embedding_ts_generation.ipynb 05_sentence_embedding_ts_generation.ipynb
papermill 06_sentiment_ts_generation.ipynb 06_sentiment_ts_generation.ipynb
papermill 07_dtw_evaluation.ipynb 07_dtw_evaluation.ipynb
papermill 08_classification_ds_preparation.ipynb 08_classification_ds_preparation.ipynb