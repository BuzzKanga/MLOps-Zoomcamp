#!/usr/bin/env python
# coding: utf-8

import os
import sys
import boto3
import pickle
import pandas as pd
from datetime import datetime


def read_data(filename):
    df = pd.read_parquet(filename)
    return df

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)   

def prepare_data(df_data):

    df_data['duration'] = df_data.tpep_dropoff_datetime - df_data.tpep_pickup_datetime
    df_data['duration'] = df_data.duration.dt.total_seconds() / 60

    df_data = df_data[(df_data.duration >= 1) & (df_data.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    # df_data[categorical] = df_data[categorical].fillna(-1).astype('float')
    df_data[categorical] = df_data[categorical].fillna(-1).astype('int').astype('str')

    return df_data


def main(raw_data):
    # dir = "output"
    # if not os.path.exists(dir):
    #     os.mkdir(dir)
    # output_file = f'output/predictions.parquet'

    # aws_access_key_id = 'your-access-key-id'
    # aws_secret_access_key = 'your-secret-access-key'
    # aws_region = 'ap-southeast-2'
    
    s3_bucket_name = 'mlopszoomcamp'
    s3_output_key = 'predictions.parquet'
    categorical = ['PULocationID', 'DOLocationID']

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    raw_df = pd.DataFrame(raw_data, columns=columns)
    df = prepare_data(raw_df)
    df['ride_id'] = df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())
    print('predicted sum:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    # df_result.to_parquet(output_file, engine='pyarrow', index=False)

    # Save the DataFrame to a local file
    local_output_file = 'predictions.parquet'
    df_result.to_parquet(local_output_file, engine='pyarrow', index=False)

    # Upload the local file to S3 bucket

    # session = boto3.Session(
    #     aws_access_key_id=aws_access_key_id,
    #     aws_secret_access_key=aws_secret_access_key,
    #     region_name=aws_region
    # )

    s3_client = boto3.client('s3')
    s3_client.upload_file(local_output_file, s3_bucket_name, s3_output_key)

    print('Output file uploaded to S3 bucket:', s3_bucket_name)

if __name__ == '__main__':
    predict_data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    main(predict_data)
