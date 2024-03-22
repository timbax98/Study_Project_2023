# region Setup

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
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
        'x': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    context = tf.io.parse_tensor(example['context'], out_type=tf.float32)
    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    target = tf.io.parse_tensor(example['target'], out_type=tf.float32)

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
        transformer.save("weights/transformer", save_format="tf")
        # tf.saved_model.save(transformer, export_dir="transformer")
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


def positional_encoding(max_seq_len, d_model):
        pos = tf.cast(tf.range(max_seq_len)[:, tf.newaxis], tf.float32)
        i = tf.range(d_model)[tf.newaxis, :]
        angle_rates = tf.cast(1 / tf.pow(10000, (2 * (i // 2)) / d_model), tf.float32)
        angle_rads = pos * angle_rates

        # Apply sine to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Apply cosine to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...] # add batch axis?

        return tf.cast(pos_encoding, dtype=tf.float32)


def normalize(inputs):
    # normalize into range [-pi, pi]
    min_val = tf.reduce_min(inputs)
    max_val = tf.reduce_max(inputs)
    normalized_inputs = tf.subtract(tf.divide(tf.subtract(inputs, min_val),
                                              tf.subtract(max_val, min_val)), 0.5) * 2 * np.pi
    return normalized_inputs


def sine_embedding(inputs, vocab_size, embedding_dim=8):
    # Initialize an empty list to store embeddings
    embeddings_list = []
    
    # Calculate sine and cosine embeddings for each value in inputs
    for value in range(vocab_size):
        value_embedding = []
        # Generate embedding components for each dimension
        for i in range(embedding_dim // 2):
            sine_embeddings = tf.sin((i + 1) * inputs)
            cosine_embeddings = tf.cos((i + 1) * inputs)
            value_embedding.extend([sine_embeddings, cosine_embeddings])
        
        # Stack embeddings for each dimension
        value_embedding = tf.stack(value_embedding, axis=-1)
        embeddings_list.append(value_embedding)
    
    # Concatenate embeddings along the last axis to create the final embedding
    final_embedding = tf.stack(embeddings_list, axis=0)
    
    return final_embedding

# endregion
    
# region Model Architecture

# Embedding + Positional Encoding
class PositionalEmbeddingContext(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.normalization = normalize # function reference
        self.pos_encoding = positional_encoding(MAX_SEQ_LEN, d_model)

    def call(self, inputs):
        inputs = self.normalization(inputs)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs += self.pos_encoding[:, :tf.shape(inputs)[1], :]
        return inputs


class PositionalEmbeddingInput(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, d_model):
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.d_model = d_model
        self.normalization = normalize # function reference
        self.embedding = sine_embedding
        self.pos_encoding = positional_encoding(MAX_SEQ_LEN, d_model)

    def call(self, inputs):
        inputs = self.normalization(inputs)
        inputs = self.embedding(inputs, self.input_vocab_size, self.d_model) # shape (batch_size, seq_len, 8)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs += self.pos_encoding[:, :tf.shape(inputs)[1], :]
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
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
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

    def call(self, x):
        x = self.self_attention(x)
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

    def call(self, x, training):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


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

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.

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

    def call(self, x, context, training):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
            
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

    def call(self, inputs, training):
        
        context, x = inputs

        print(f"context shape is {context.shape}.")
        print(f"x shape is {x.shape}.")

        context = self.encoder(context, training)
        x = self.decoder(x, context, training)

        print(f"encoder output shape is {context.shape}.")
        print(f"decoder output shape is {x.shape}.")

        logits = self.final_layer(x)
        print(f"logits shape is {logits.shape}.")

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            print("Did not delete keras mask.")
            pass

        print(f"output shape is {logits.shape}.")
        return logits

    #@tf.function
    def train_step(self, inputs, targets):

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss_function(targets, predictions)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss
    
    def predict(self, inputs, targets):
        
        predictions = self(inputs, training=False)
        loss = self.loss_function(targets, predictions)

        return loss, predictions

# endregion

# region Data
    
# Import dataset from path
import_path = "./data/mock_ds.zip"

# Create a TFRecordDataset from the saved file and deserialize
dataset = tf.data.TFRecordDataset(import_path, compression_type='GZIP')
dataset = dataset.map(deserialize)

# Debugging: check train dataset
"""
samples_to_print = 1
print(f"\n \n Printing {samples_to_print} samples")
for batch, (con, x, target) in enumerate(dataset.take(samples_to_print)):
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
num_batches = 0
for batch in dataset:
    num_batches += 1

train_size = int(0.8 * num_batches) # roughly 80% train split
test_size = num_batches - train_size

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size).take(test_size)

# endregion

# region Training

### Hyperparameters

# Model HP
num_layers = 4 # How often to stack encoder & decoder blocks
d_model = 8 # Embedding Dimensionality (128) - might need to use 8 here, because our embedded input also only has 8 dims.
dff = 512 # Dimensionality of the feed-forward sublayer
num_heads = 8 # Number of parallel self attention heads in the MHA layer
dropout_rate = 0.1

# Data HP
input_vocab_size = 364 # number of tokens of input, i.e. angles
batch_size = max(con.shape[0] for con, x, t in dataset)
MAX_SEQ_LEN = max(con.shape[1] for con, x, t in dataset)
print(f"\nNumber of batches: {num_batches}, Batch size: {batch_size}, Maximum sequence length: {MAX_SEQ_LEN}\n")

# Training HP
EPOCHS = 100
PATIENCE = 100
LEARNING_RATE = 0.001

# Instantiate the Transformer model and print summary.
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)

# Build the model using first batch
for con, x, target in train_ds.take(1):
    break

output = transformer((con, x), training=False)

transformer.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#transformer.summary()

### Model training

# Logging
config_name= f"epochs{EPOCHS}lr{LEARNING_RATE}p{PATIENCE}"
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
        loss = transformer.train_step((context, x), target)
        loss_metric_train.update_state(values=loss)

    print(f"Train loss at epoch {epoch+1}: {loss_metric_train.result()}")

    # Logging train metrics
    with train_summary_writer.as_default():
        tf.summary.scalar(f"{loss_metric_train.name}", loss_metric_train.result(), step=epoch)

    # Reset metrics
    loss_metric_train.reset_states()


    # Test loop
    for batch, (context, x, target) in enumerate(test_ds):

        # Get prediction and calculate loss
        loss, prediction_ = transformer.predict((context, x), target)
        loss_metric_test.update_state(values=loss)

    # Append test accuracy and save the model
    test_losses.append(loss_metric_test.result())
    info = save_weights(loss_metric_test.result(), epoch, transformer)

    # Logging test metrics
    with test_summary_writer.as_default():
        tf.summary.scalar(f"{loss_metric_test.name}", loss_metric_test.result(), step=epoch)

    print(f"Test loss at epoch {epoch+1}: {loss_metric_test.result()}")

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}\nBatch {batch+1}/{test_size}\n\nTarget: {target.shape}\n{target}\n\nPrediction: {prediction_.shape}\n{prediction_}\n\n")

    # Early stopping
    if early_stopping(loss_metric_test.result()):
      print("Training stopped")
      break

    # Reset
    loss_metric_test.reset_states()

# endregion