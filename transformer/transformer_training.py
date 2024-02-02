#!/usr/bin/env python
# coding: utf-8

# # **Study Project:** *Transformer model for prediction of grasping movements*

# ## Import packages

# In[1]:

''''''

import os
import zipfile
import numpy as np
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.utils import Progbar
#from tensorflow.keras.callbacks import TensorBoard

# Model/Training Hyperparamters
EPOCHS = 500
LEARNING_RATE = 0.001 ## not used yet
PATIENCE = 10

# ## Helper functions


# In[2]:


# Model checkpointing
def save_weights(current_loss,epoch,transformer):

  # If currernt test_accuracy is the highest
  if all(current_loss <= old_losses for old_losses in test_losses) or epoch == 0:

    # Save current model weight
    transformer.save_weights(save_path)
    return "\nWeights saved!"

  else:
    return ""


# In[3]:


def early_stopping(current_loss):

  global p_counter

  # If current loss did decrease over the last patience epochs
  if all(current_loss <= old_losses for old_losses in test_losses):
    p_counter = 0
    return False

  # Stop training and load old weights
  else:
    print(f"Patience: {p_counter+1}\n")
    p_counter += 1

    if p_counter == PATIENCE:
      transformer.load_weights(save_path)
      return True

    else:
      return False


# ## Model Architecture

# ### Positional Encoding

# In[4]:


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.get_positional_encoding(position, d_model)

    def get_positional_encoding(self, sequence_length, input_dim):
        angle_rads = self.get_angles(tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis],
                                     tf.range(input_dim, dtype=tf.float32)[tf.newaxis, :],
                                     input_dim)

        # Apply sine to even indices in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Apply cosine to odd indices in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Concatenate sines and cosines
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, sequence_length, i, input_dim):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(input_dim, tf.float32))
        return sequence_length * angle_rates

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]


# ### Encoder

# In[5]:


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dense = Dense(d_model)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = Dropout(rate)

    def call(self, x, training):
        x = self.dense(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x


# ### Decoder

# In[6]:


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,sequence_length, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.masked_mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.ffn = keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.look_ahead_mask = tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def generate_mask(self, size):
        # Create a lower triangular mask
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(self, inputs, enc_output, training):
        
        sequence_length = tf.shape(inputs)[1]

        # Generate a dynamic look-ahead mask
        self.look_ahead_mask = self.generate_mask(sequence_length)

        attn1 = self.masked_mha1(inputs, inputs, attention_mask=self.look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)

        attn2 = self.mha2(out1, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, output_dim, sequence_length, rate=0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff,sequence_length, rate) for _ in range(num_layers)]

        self.final_layer = Dense(output_dim)

    def call(self, x, enc_output, training):
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training)

        x = self.final_layer(x)

        return x


# ### Transformer Model

# In[7]:


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, sequence_length, input_dim, output_dim, rate=0.1):
        super(Transformer, self).__init__()

        self.positional_encoder = PositionalEncoding(sequence_length, input_dim)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, input_dim, rate)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, output_dim, sequence_length, rate)

        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, inp, training):
        x = self.positional_encoder(inp)
        enc_output = self.encoder(x, training)
        dec_output = self.decoder(x, enc_output, training)

        return dec_output

    #@tf.function
    def train_step(self, x, targets):

        with tf.GradientTape() as tape:
            predictions = self(x, training = True)
            loss = self.loss_function(targets, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def predict(self, x, targets):

        predictions = self(x,training = False)
        loss = self.loss_function(targets, predictions)

        return loss, predictions


# ## Training

# ### Import dataset

# In[8]:


# Define the file path where the zipped dataset was saved
import_path = "./data/train_ds.zip"

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
train_ds = dataset.map(deserialize_example)
test_ds = train_ds # mock example for training


# In[9]:


# Debugging
"""
# Print at most the first 5 examples from the train dataset
for i, (train_data, label_data) in enumerate(train_ds):
    if i < 5:
        print(f"Training pair: {i}")
        print("Train Data:")
        print(train_data.shape)
        print(train_data.numpy())
        print()
        print("Label Data:")
        print(label_data.shape)
        print(label_data.numpy())
        print()
"""


# ### Build model

# In[10]:

# Define hyperparameters for transformer model.
# data
batch_size = max(seq.shape[0] for seq, t in train_ds)
max_sequence_length = max(seq.shape[1] for seq, t in train_ds)
print(f"Batch size: {batch_size}, Maximum sequence length: {max_sequence_length}\n")

# model
num_layers = 4 # How often to stack encoder & decoder blocks
num_heads = 8 # Number of parallel self attention heads in the MHA layer
dff = 512 # Dimensionality of the feed-forward sublayer
input_dim = 8  # Four coordinates each for hand and object
output_dim = 1  # Output radian angle via sin and cos
d_model = input_dim # Dimensionality of the model's hidden states and the size of the model's embedding vectors


# In[11]:


# Instantiate the Transformer model and print summary.
transformer = Transformer(num_layers, d_model, num_heads, dff, max_sequence_length, input_dim, output_dim)
transformer.build([batch_size, max_sequence_length, input_dim])
transformer.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
transformer.summary()


# ### Train model

# In[12]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[13]:


# In[14]:


# Logging
config_name= "epochs500lr0001p10"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_path = f"logs/{config_name}/{current_time}/train"
test_log_path = f"logs/{config_name}/{current_time}/test"

# Log writers
train_summary_writer = tf.summary.create_file_writer(train_log_path)
test_summary_writer = tf.summary.create_file_writer(test_log_path)

# Metrics
loss_metric_train = tf.keras.metrics.Mean(name="loss_train")
loss_metric_test = tf.keras.metrics.Mean(name="loss_test")


# In[15]:


# Miscellaneous needed variables
test_losses = []
save_path = f"/weights/checkpoints/weights_{current_time}.h5"

# Main training loop
for epoch in range(EPOCHS):

    #pb_i = Progbar(len(train_ds))

    # Train loop
    for input, target_ in train_ds:
        # Train
        loss = transformer.train_step(input,target_)
        # Loss
        loss_metric_train.update_state(values=loss)
        # Update progbar
        #pb_i.add(1)

    print(f"Train loss at epoch {epoch+1}: {loss_metric_train.result()}")

    # Logging train metrics
    with train_summary_writer.as_default():
        tf.summary.scalar(f"{loss_metric_train.name}", loss_metric_train.result(), step=epoch)

    # Reset metrics
    loss_metric_train.reset_states()

    # Test loop
    for input,target_ in test_ds:
        # Get predictions
        loss, prediction_ = transformer.predict(input,target_)
        # Loss
        loss_metric_test.update_state(values=loss)

    # Append test_acc
    test_losses.append(loss_metric_train.result())

    # Checkpointing
    info = save_weights(loss_metric_train.result(),epoch,transformer)

    # Logging test metrics
    with test_summary_writer.as_default():
        tf.summary.scalar(f"{loss_metric_test.name}", loss_metric_test.result(), step=epoch)

    print(f"Test loss at epoch {epoch+1}: {loss_metric_test.result()}")

    # Early stopping
    if early_stopping(loss_metric_train.result()):
      print("Training stopped")
      break

    # Reset
    loss_metric_test.reset_states()

# Save the model
transformer.save("model", save_format="tf")


# In[16]:


# Visualize training process
#get_ipython().run_line_magic('tensorboard', '--logdir=logs/')

