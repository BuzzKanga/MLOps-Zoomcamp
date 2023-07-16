import pandas as pd
from pandas.testing import assert_frame_equal
from datetime import datetime
import batch

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)   

def test_prepare_data():
    # data setup

    test_data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    test_df = pd.DataFrame(test_data, columns=columns)

    expected_data = [
        (-1.0, -1.0, dt(1, 2), dt(1, 10), 8.0),
        (1.0, -1.0, dt(1, 2), dt(1, 10), 8.0),
        (1.0, 2.0, dt(2, 2), dt(2, 3), 1.0),
    ]
    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    # expected_df['PULocationID'] = expected_df['PULocationID'].astype('str')
    # expected_df['DOLocationID'] = expected_df['DOLocationID'].astype('str')

    # check prepare_data()
    actual_df = batch.prepare_data(test_df)

    assert actual_df.equals(expected_df) 