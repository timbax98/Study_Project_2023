#!/usr/bin/env python
# coding: utf-8

# # **Study Project:** *Transformer model for prediction of grasping movements*

# ## Import packages

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


# ## Prediction

# ### Import trained model

# In[2]:


transformer = load_model("model")


# ### Generate predictions

# In[3]:

# Define the file path where the zipped dataset was saved
import_path = "./data/test_ds.zip"

# Function to deserialize tensors from bytes
def deserialize_example(serialized_example):
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    y = tf.io.parse_tensor(example['y'], out_type=tf.float32)

    return x, y

# Create a TFRecordDataset from the saved file
dataset = tf.data.TFRecordDataset(import_path, compression_type='GZIP')

# Deserialize the zipped dataset
test_ds = dataset.map(deserialize_example)


# Obtain predictions
predictions = transformer.predict(test_ds)
#print(f"Shape: {predictions.shape}, \nPredictions: \n{predictions}")
print(f"Shape: {predictions.shape}")

# surrogate predictions from train ds
predictions_ds = test_ds.map(lambda train_data, label_data: (train_data, predictions))

# Debugging: check dataset
"""
samples_to_print = 1
print("\n \n Printing {} predictions_ds".format(samples_to_print))
for i, (train_data, prediction_data) in enumerate(predictions_ds.take(samples_to_print)):
    if i < 5: # print at most the first 5 examples
        print(f"Training pair: {i}")
        print("Train Data:")
        print(train_data.shape)
        print(train_data.numpy())
        print()
        print("Prediction:")
        print(prediction_data.shape)
        print(prediction_data.numpy())
        print()
"""


# Visualize results
plt.figure(figsize=(8, 8))
plt.xlim(0, 3000)
plt.ylim(0, 3000)

for batch_i, (sample, prediction) in enumerate(predictions_ds):
    for vid in sample:
        print(f"vid: {vid}")
        vid = vid[0]
        bboxes = vid[np.all((vid != -999) & (vid != -333))]
        bboxes_hands = bboxes[:, 4:6]
        print(bboxes_hands)
        break
