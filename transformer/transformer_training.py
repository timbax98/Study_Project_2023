"""
Transformer Model for Language Understanding

This Python script contains an adaptation of the Transformer model for language understanding
tutorial provided by TensorFlow. The original tutorial can be found at:
https://www.tensorflow.org/text/tutorials/transformer

Author: TensorFlow Authors
Title: Neural machine translation with a Transformer and Keras
Website: TensorFlow official website
URL: https://www.tensorflow.org/text/tutorials/transformer
Year: 2024-03-23 UTC

The code in this script is adapted from the TensorFlow tutorial mentioned above.

This script is distributed under the Creative Commons Attribution 4.0 License:
https://creativecommons.org/licenses/by/4.0/
"""

# region Setup

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import shutil
import tensorflow as tf
import datetime
import zipfile

# endregion

# region Helpers

def deserialize(serialized_example):
    """
    Function to deserialize tensors from bytes.
    """

    feature_description = {
        'context': tf.io.FixedLenFeature([], tf.string),
        'input': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    context = tf.io.parse_tensor(example['context'], out_type=tf.float64)
    x = tf.io.parse_tensor(example['input'], out_type=tf.float64)
    target = tf.io.parse_tensor(example['target'], out_type=tf.float64)

    return context, x, target


def save_weights(current_loss,epoch,transformer):
    """
    Model checkpointing.
    """

    # If currernt test_accuracy is the highest
    if all(current_loss <= old_losses for old_losses in test_losses) or epoch == 0:

        # Remove the existing model directory
        shutil.rmtree("weights/transformer", ignore_errors=True)

        # Save current model
        tf.keras.models.save_model(transformer, "models/transformer", save_format="tf") # easier export/import
        #tf.saved_model.save(transformer, export_dir="transformer") # more flexibility and compatibility
        return "\nModel saved!"

    else:
        return ""


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
      return True

    else:
      return False


def compute_mask(inputs, padding_token=-2):
    return tf.cast(tf.not_equal(inputs, padding_token), tf.float64)


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)
    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    return pos_encoding


def normalize(inputs, lower_bound=0, upper_bound=1, min_val=None, max_val=None):
    # If min_val and max_val are not provided, compute them from the inputs
    if min_val is None:
        min_val = tf.reduce_min(inputs)
    if max_val is None:
        max_val = tf.reduce_max(inputs)

    try:
        if lower_bound >= upper_bound:
            raise ValueError(f'Lower bound (current: {lower_bound}) must be smaller than upper bound (current: {upper_bound}).')

        # Scale inputs to the custom range [lower_bound, upper_bound]
        scaled_inputs = lower_bound + (inputs - min_val) * (upper_bound - lower_bound) / (max_val - min_val)

        return scaled_inputs

    except ValueError as e:
        print("Error:", e)


def sine_embedding(inputs, d_model):
    # Apply sine and cosine transformations to each value
    sine_embeddings = tf.sin(inputs)
    cosine_embeddings = tf.cos(inputs)
    
    # Stack sine and cosine embeddings together into ([sin,cos,sin,cos,sin,cos,sin,cos])
    alternating_embeddings = tf.stack([sine_embeddings, cosine_embeddings] * (d_model//2), axis=-1)
    
    # Stack sine and cosine embeddings together into ([sin,sin,sin,sin,cos,cos,cos,cos])
    repeated_embeddings = tf.stack([sine_embeddings, cosine_embeddings], axis=-1)
    repeated_embeddings = tf.repeat(repeated_embeddings, repeats=d_model//2, axis=-1)
    
    return alternating_embeddings


def masked_loss(label, pred, pad_token=-2):
    mask = label != pad_token
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred, pad_token=-2):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != pad_token

    match = match & mask

    match = tf.cast(match, dtype=tf.float64)
    mask = tf.cast(mask, dtype=tf.float64)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float64)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float64)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }

# endregion
    
# region Model Architecture

# Embedding + Positional Encoding
class PositionalEmbeddingContext(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.normalization = normalize # function reference
        self.pos_encoding = positional_encoding(MAX_SEQ_LEN, depth=d_model)

    def call(self, inputs):

        seq_len = tf.shape(inputs)[1]
        #print(f'PosEmbedding Context (raw)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        inputs = self.normalization(inputs, lower_bound=-1, upper_bound=1, min_val=0, max_val=1920)
        #print(f'PosEmbedding Context (normalized)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        # This factor sets the relative scale of the embedding and positonal_encoding.
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        #print(f'PosEmbedding Context (scaled)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        inputs += self.pos_encoding[tf.newaxis, :seq_len, :] # add batch axis, take pos encoding for positions up to sequence length and all embedding dimensions.
        #print(f'PosEncoding\n{self.pos_encoding.shape}, {self.pos_encoding.dtype}\n{self.pos_encoding}\n\n')
        #print(f'PosEmbedding Context (pos encoded)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        return inputs


class PositionalEmbeddingInput(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, d_model):
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.d_model = d_model
        self.normalization = normalize # function reference
        self.embedding = sine_embedding
        self.pos_encoding = positional_encoding(MAX_SEQ_LEN, depth=d_model)

    def call(self, inputs):

        seq_len = tf.shape(inputs)[1]
        inputs = tf.squeeze(inputs, axis=-1) # strip last dimension
        #print(f'PosEmbeddingInput (raw)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        inputs = self.normalization(inputs, lower_bound=-np.pi, upper_bound=np.pi, min_val=0, max_val=360)
        #print(f'PosEmbeddingInput (normalized)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        inputs = self.embedding(inputs, self.d_model) # shape (batch_size, seq_len, 8)
        # Manual embedding of input SOS token (first row of embedding in each video in the batch is set to 4x (0,0), as this is mid of circle)
        mask = tf.concat([tf.zeros_like(inputs[:, :1, :]), tf.ones_like(inputs[:, 1:, :])], axis=1)
        inputs *= mask
        #print(f'PosEmbeddingInput (embedded)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        # This factor sets the relative scale of the embedding and positonal_encoding.
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        #print(f'PosEmbeddingInput (scaled)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        inputs += self.pos_encoding[tf.newaxis, :seq_len, :]
        #print(f'PosEncoding\n{self.pos_encoding.shape}, {self.pos_encoding.dtype}\n{self.pos_encoding}\n\n')
        #print(f'PosEmbeddingInput (pos encoded)\n{inputs.shape}, {inputs.dtype}\n{inputs[0]}\n\n')

        return inputs


# Attention Layers
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x, mask):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=mask) # Mask padding tokens from Encoder input
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x, mask):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=mask, # Mask padding tokens from Decoder input
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# Feed-Forward Layer
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x


# Encoder
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, mask):
        x = self.self_attention(x, mask)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    """ Encoder module """
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbeddingContext(d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):

        x = self.pos_embedding(x)  # shape (batch_size, seq_len, d_model)
        embedded_x = x # comparison dummy

        # Add dropout.
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        #print(f'Encoder Output\n{x.shape}, {x.dtype}\n{x[0]}\n\n')
        return x  # shape (batch_size, seq_len, d_model)


# Decoder
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context, mask):
        x = self.causal_self_attention(x=x, mask=mask)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x) # shape (batch_size, seq_len, d_model)

        return x
    

class Decoder(tf.keras.layers.Layer):
    """ Decoder module """
    def __init__(self, num_layers, d_model, num_heads, dff, output_dim, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbeddingInput(input_vocab_size=output_dim, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                        dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context, training, mask):
        
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        embedded_x = x # comparison dummy

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context, mask)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        
        #print(f'Decoder Output\n{x.shape}, {x.dtype}\n{x[0]}\n\n')
        return x # (batch_size, target_seq_len, d_model)


# Transformer Model
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate) # delete vocab size only in encoder or also decoder? needed for embedding?
        self.final_layer = tf.keras.layers.Dense(input_vocab_size)

        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, inputs, mask=None, training=False):

        context, x = inputs

        context = self.encoder(context, training, mask)
        x = self.decoder(x, context, training, mask)

        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            #print("Did not delete keras mask.")
            pass

        #print(f'Transformer Output (logits)\n{logits.shape}, {logits.dtype}\n{logits[0]}\n\n')
        return logits

    #@tf.function
    def train_step(self, inputs, targets, mask):

        with tf.GradientTape() as tape:
            predictions = self(inputs, mask, training=True)
            # mask padding tokens in targets
            targets *= mask
            loss = self.loss_function(targets, predictions)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss
    
    def predict(self, inputs, targets, mask):
        
        predictions = self(inputs, mask, training=False)
        # mask padding tokens in targets
        targets *= mask
        loss = self.loss_function(targets, predictions)

        return loss, predictions

# endregion

# region Data
    
# Import dataset from path
train_path = "./data/train_ds.zip"
test_path = "./data/test_ds.zip"

# Create a TFRecordDataset from the saved file and deserialize
train_ds = tf.data.TFRecordDataset(train_path, compression_type='GZIP')
train_ds = train_ds.map(deserialize)
test_ds = tf.data.TFRecordDataset(test_path, compression_type='GZIP')
test_ds = test_ds.map(deserialize)

# Debugging: check train dataset
"""
samples_to_print = 1
print(f"\n \n Printing {samples_to_print} samples")
for batch, (con, x, target) in enumerate(train_ds.take(samples_to_print)):
    if batch < 5: # print at most the first 5 examples
        print(f"Batch: {batch}")
        print("Context:")
        print(con.numpy().shape)
        print(con.numpy())
        print()
        print("x:")
        print(x.numpy().shape)
        print(x.numpy())
        print()
        print("target:")
        print(target.numpy().shape)
        print(target.numpy())
        print()
"""

# Split dataset into training and validation
num_batches_train = 0
for batch in train_ds:
    num_batches_train += 1

num_batches_test = 0
for batch in test_ds:
    num_batches_test += 1

#print(f"Train shape: {len(train_ds)}, \ntest shape: {len(test_ds)}")
print(f"Train shape: {num_batches_train}, \ntest shape: {num_batches_test}")

# endregion

# region Training

### Hyperparameters

# Model HP
num_layers = 4 # How often to stack encoder & decoder blocks
d_model = 8 # Embedding Dimensionality (128) - might need to use 8 here, because our embedded input also only has 8 dims.
dff = 32 # Dimensionality of the feed-forward sublayer (512)
num_heads = 8 # Number of parallel self attention heads in the MHA layer
dropout_rate = 0.1

# Data HP
input_vocab_size = 363 # number of tokens of input, i.e. angles
batch_size = max(con.shape[0] for con, x, t in train_ds)
max_train = max(con.shape[1] for con, x, t in train_ds)
max_test = max(con.shape[1] for con, x, t in test_ds)
MAX_SEQ_LEN = max(max_train, max_test)
PAD = -2
print(f"\nTrainDS: number of batches: {num_batches_train}, batch size: {batch_size}, maximum sequence length: {max_train}")
print(f"\nTestDS: number of batches: {num_batches_test}, batch size: {batch_size}, maximum sequence length: {max_test}\n")

# Training HP
EPOCHS = 3000
PATIENCE = 100
LEARNING_RATE = 0.001 # not used yet

"""
# Debugging: PosEncoding visualization
pos_encoding = positional_encoding(length=MAX_SEQ_LEN, depth=d_model)

# Check the shape.
print(pos_encoding.shape)

# Plot the dimensions.
plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()
"""

# Instantiate the Transformer model
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)

# Build the model using first batch
for con, x, target in train_ds.take(1):
    break

output = transformer((con, x))

# Setup optimizer with custom learning rate schedule
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# Compile the model
transformer.compile(
    loss=masked_loss,
    optimizer="adam", # does currently not use custom lr schedule
    metrics=[masked_accuracy])

#transformer.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#transformer.summary()

### Model training

# Logging
config_name= f"epochs{EPOCHS}p{PATIENCE}"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_path = f"logs/{config_name}/{current_time}/train"
test_log_path = f"logs/{config_name}/{current_time}/test"
train_summary_writer = tf.summary.create_file_writer(train_log_path)
test_summary_writer = tf.summary.create_file_writer(test_log_path)
# Metrics
loss_metric_train = tf.keras.metrics.Mean(name="loss_train")
loss_metric_test = tf.keras.metrics.Mean(name="loss_test")
test_losses = []


for epoch in range(EPOCHS):

    # Training loop
    for batch, (context, x, target) in enumerate(train_ds):

        # Perform one training step and compute its loss
        mask = compute_mask(target, padding_token=PAD)
        loss = transformer.train_step((context, x), target, mask)
        loss_metric_train.update_state(values=loss)
        #print(f"Training... Epoch {epoch+1}, Batch {batch+1}/{num_batches_train}.")

    print(f"Train loss at epoch {epoch+1}: {loss_metric_train.result()}")

    # Logging train metrics
    with train_summary_writer.as_default():
        tf.summary.scalar(f"{loss_metric_train.name}", loss_metric_train.result(), step=epoch)

    # Reset metrics
    loss_metric_train.reset_states()


    # Test loop
    for batch, (context, x, target) in enumerate(test_ds):

        # Get prediction and calculate loss
        mask = compute_mask(target, padding_token=PAD)
        loss, prediction_ = transformer.predict((context, x), target, mask)
        loss_metric_test.update_state(values=loss)

    # Append test accuracy and save the model
    test_losses.append(loss_metric_test.result())
    info = save_weights(loss_metric_test.result(), epoch, transformer)

    # Logging test metrics
    with test_summary_writer.as_default():
        tf.summary.scalar(f"{loss_metric_test.name}", loss_metric_test.result(), step=epoch)

    print(f"Test loss at epoch {epoch+1}: {loss_metric_test.result()}")

    # Early stopping
    if early_stopping(loss_metric_test.result()):
      print("Training stopped")
      break

    # Reset
    loss_metric_test.reset_states()

# endregion