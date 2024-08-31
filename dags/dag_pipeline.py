"""
# Data Pipeline
This DAG performs the ingest and transformation of a dataset in csv format, stores the intermediate steps and
train a classification model based.
The model is stored as well and used to evaluate its performance based on the accuracy metric.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import includes.process_data as dpipe
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta



# Get Dags directory path
DAG_PATH = os.getcwd()
RAW_DATA_FILE = f'{DAG_PATH}/data/raw/dataset.csv'
PROCESSED_DATA_PATH = f'{DAG_PATH}/data/processed/'
FEATURE_SELECTED_FILES = f'{PROCESSED_DATA_PATH}/dataset_features_selected.pkl'

# List with the features to select
features_selected = ['rev_Mean', 'eqpdays', 'drop_blk_Mean', 'uniqsubs', 'actvsubs',
                     'custcare_Mean', 'complete_Mean', 'attempt_Mean', 'hnd_price',
                     'months', 'mou_Mean', 'totcalls', 'totrev', 'asl_flag', 'dualband',
                     'hnd_webcap', 'refurb_new', 'marital', 'models', 'adults', 'kid0_2',
                     'kid3_5', 'kid6_10', 'kid11_15', 'kid16_17'
                     ]

# Target variable
target_var = "churn"

# Define DAG parameters
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize DAG
with DAG(
    dag_id='dag_pipeline',
    schedule_interval=None,  # timedelta(days=1),  # Set your desired schedule
    catchup=False,
    doc_md=__doc__,
    description='Dataflow for training a ML algorithm on a telco company dataset.',
    tags=['python', 'telco', 'classifier'],
    default_args=default_args
) as dag:

    # Load data and select variables (features + target)
    data_load_feat_selection = PythonOperator(
        task_id='data_load_and_feature_selection',
        python_callable=dpipe.feature_selection,
        op_kwargs={'features_selected': [*features_selected, target_var],
                   'input_path': RAW_DATA_FILE,
                   'output_path': PROCESSED_DATA_PATH,
                   'sep': ";",
                   'decimal': ','}
    )

    # Split data into train and test sets
    split_train_test = PythonOperator(
        task_id='split_tran_test',
        python_callable=dpipe.split_data,
        op_kwargs={'input_path': FEATURE_SELECTED_FILES,
                   'output_path': PROCESSED_DATA_PATH,
                   'target_feature': target_var,
                   'test_size': 0.2}
    )

    # Clean train and test sets
    clean_train_test = PythonOperator(
        task_id='clean_train_test',
        python_callable=dpipe.clean_data,
        op_kwargs={'train_input_path': f"{PROCESSED_DATA_PATH}/train_features.pkl",
                   'test_input_path': f"{PROCESSED_DATA_PATH}/test_features.pkl",
                   'output_path': PROCESSED_DATA_PATH,
                   'drop_columns_bool': False,
                   'missing_val_threshold': 0.1,
                   'scale_data': True,
                   'scaler_type': 'standard'
                   }
    )

    # Encode categorical features
    encode_train_test = PythonOperator(
        task_id='encode_train_test',
        python_callable=dpipe.encode_data,
        op_kwargs={'train_input_path': f"{PROCESSED_DATA_PATH}/train_features_cleaned.pkl",
                   'test_input_path': f"{PROCESSED_DATA_PATH}/test_features_cleaned.pkl",
                   'output_path': PROCESSED_DATA_PATH,
                   'encoder_type': 'onehot'
                   }
    )

    # Train model
    train_model_classifier = PythonOperator(
        task_id='train_model_classifier',
        python_callable=dpipe.train_model_classification,
        op_kwargs={'features_train_input_path': f"{PROCESSED_DATA_PATH}/train_features_encoded.pkl",
                   'target_train_input_path': f"{PROCESSED_DATA_PATH}/train_target.pkl",
                   'output_path': PROCESSED_DATA_PATH
                   }
    )

    # Evaluate model
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=dpipe.evaluate_model,
        op_kwargs={'features_test_input_path': f"{PROCESSED_DATA_PATH}/test_features_encoded.pkl",
                   'target_test_input_path': f"{PROCESSED_DATA_PATH}/test_target.pkl",
                   'model_path': f"{PROCESSED_DATA_PATH}/trained_rfc_model.pkl"
                   }
    )

    # Task dependencies and order
    data_load_feat_selection >> split_train_test >> clean_train_test >> encode_train_test >> train_model_classifier >> evaluate_model
