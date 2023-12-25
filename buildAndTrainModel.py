

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import customImputerLayerDefinition as myImputer
from customImputerLayerDefinition import myImputer as myImputer

# Define the feature description
feature_description = {
    'tickers': tf.io.FixedLenFeature([188], tf.float32, np.zeros(188)),
    'weekday': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'month': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'hour': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'target': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}

# Parse the serialized examples into a dictionary of tensors
def parse_example(serialized_example):
    example = tf.io.parse_example(serialized_example, feature_description)
    features = {k: v for k, v in example.items() if k != 'target'}
    target = example['target']
    return features, target

# Split the dataset into training, validation, and testing datasets
raw_dataset = tf.data.TFRecordDataset(['dataset.tfrecord'])
datLen = raw_dataset.reduce(0,lambda x,y: x+1)
n_valid = int(datLen.numpy()*.1)
n_test = int(datLen.numpy()*.1)
n_train = datLen.numpy()-n_valid-n_test
train = raw_dataset.take(n_train).batch(2048).map(
    parse_example,num_parallel_calls=8).cache()

test = raw_dataset.skip(n_train).take(n_test).batch(2048).map(
    parse_example,num_parallel_calls=8).cache()

valid = raw_dataset.skip(n_train+n_test).take(n_valid).batch(2048).map(
    parse_example,num_parallel_calls=8).cache()

# Define input layers
inputDict = {
    'tickers': tf.keras.Input(shape=(188,), dtype=tf.float32),
    'weekday': tf.keras.Input(shape=(), dtype=tf.int64),
    'month': tf.keras.Input(shape=(), dtype=tf.int64),
    'hour': tf.keras.Input(shape=(), dtype=tf.int64)
}

# Create an instance of the Imputer layer
imputer = myImputer()
imputer.adapt(train.map(lambda x,y: x['tickers']))
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train.map(lambda x,y: imputer(x['tickers'])))

weekday_nTokens = 6
weekday_catEncoder=tf.keras.layers.IntegerLookup(max_tokens=weekday_nTokens,num_oov_indices=0)
weekday_catEncoder.adapt(train.map(lambda x,y:x['weekday']))
weekday_catInts=weekday_catEncoder(inputDict['weekday'])

month_nTokens = 12
month_catEncoder=tf.keras.layers.IntegerLookup(max_tokens=month_nTokens,num_oov_indices=0)
month_catEncoder.adapt(train.map(lambda x,y:x['month']))
month_catInts=month_catEncoder(inputDict['month'])

hour_nTokens = 24
hour_catEncoder=tf.keras.layers.IntegerLookup(max_tokens=hour_nTokens,num_oov_indices=0)
hour_catEncoder.adapt(train.map(lambda x,y:x['hour']))
hour_catInts=hour_catEncoder(inputDict['hour'])

weekday_embedding = tf.keras.layers.Embedding(weekday_nTokens, 2)(weekday_catInts)
month_embedding = tf.keras.layers.Embedding(month_nTokens, 2)(month_catInts)
hour_embedding = tf.keras.layers.Embedding(hour_nTokens, 2)(hour_catInts)

# Flatten the embedding outputs
weekday_embedding = tf.keras.layers.Flatten()(weekday_embedding)
month_embedding = tf.keras.layers.Flatten()(month_embedding)
hour_embedding = tf.keras.layers.Flatten()(hour_embedding)

# Concatenate all the inputs
preproced = tf.concat([normalizer(imputer(inputDict['tickers'])), weekday_embedding, month_embedding, hour_embedding], axis=-1)

# Creating the model
restMod = tf.keras.Sequential([
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1024,activation='relu'),

    tf.keras.layers.Dense(22, activation='softmax')
    ])

decs = restMod(preproced)
whole_model = tf.keras.Model(inputs=inputDict, outputs=decs)
whole_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
whole_model.summary()

# Defining callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('mySavedModel', save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

# Training the model
history = whole_model.fit(train, epochs=200, verbose=1, validation_data=valid, callbacks=[early_stopping_cb, checkpoint_cb])

# Evaluating the model
whole_model.evaluate(test)
whole_model.save('mySavedModel')

# Plotting the accuracy
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
