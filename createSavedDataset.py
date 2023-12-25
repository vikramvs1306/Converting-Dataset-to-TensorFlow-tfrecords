import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request

# loading the Dataset
pkl = pd.read_pickle("./data/appml-assignment1-dataset-v2.pkl")
X = pkl['X']
y = pkl['y']

# Fractional Change
change = (X['CAD-high'] - X['CAD-close']) / X['CAD-close']
bins = np.linspace(-0.001, 0.001, 21)
target = np.digitize(change, bins=bins)

date = pd.to_datetime(X['date'])

# Day of the Week
weekday = date.dt.day_of_week.astype('category')

# Hour of the Day
hour = date.dt.hour.astype('category')

# Month of the Year
month = date.dt.month.astype('category')

# Tikcers
tickers = X.iloc[:, 1:]

# Feature Creation
from tensorflow.train import FloatList,Int64List, Feature,Features,Example


with tf.io.TFRecordWriter('dataset.tfrecord') as f:
    for index in range(len(X)):
        feature={
            'tickers':Feature(float_list=FloatList(value=tickers.values[index])),
            'weekday':Feature(int64_list=Int64List(value=[weekday.values[index]])),
            'month':Feature(int64_list=Int64List(value=[month.values[index]])),
            'hour':Feature(int64_list=Int64List(value=[hour.values[index]])),
            'target':Feature(int64_list=Int64List(value=[target[index]])),
        }
        myExamp=Example(features=Features(feature=feature))
        f.write(myExamp.SerializeToString())
