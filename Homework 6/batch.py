#!/usr/bin/env python
# coding: utf-8

import sys
import boto3
import pickle
import pandas as pd
from datetime import datetime


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://mlopszoomcamp/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


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


def main(year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    categorical = ['PULocationID', 'DOLocationID']

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    raw_df = read_data(input_file)
    df = prepare_data(raw_df)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


if __name__ == '__main__':
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    main(year, month)
