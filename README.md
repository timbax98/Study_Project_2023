# Study Project: Sensory augmentation and grasping movements

This repository contains the study project on Sensory augmentation and grasping movements in SS23 & WS23/24 of the University of Osnabrück. The project builds on an OptiVisT subproject, which focuses on supporting blind individuals in guiding their hand movements using a tactile bracelet and vibration motor inputs. Here, we aim to enhance this concept by implementing a transformer neural network for hand movement guidance. Instead of simple left, right, up, and down commands from the bracelet, the transformer predicts naturalistic hand movement trajectories which are then translated into corresponding directional vibrations on an improved version of the tactile bracelet.

## Dependencies

The code relies on the following external libraries:

- **cv2**: OpenCV library for video processing.
- **numpy**: Numerical computing library for efficient array manipulation.
- **yolov5**: Ultralytics library for object detection with YOLOv5.
- **tensorflow**: An open-source machine learning framework.

Please ensure that these libraries are installed and available before running the code.

## 1. Data creation

The create_dataset.py script contains all functionality for creating input (training) data from raw videofiles for the transformer model.

### <em>Bounding Box extraction</em>

The script includes code for extracting bounding box information from videos, performing interpolation to fill missing information, and concatenating the input data for each video. The following function extracts bounding boxes from a video using an object detector (YOLOv5 models, trained on COCO and EgoHands DS).

```python
def GetBoundingBoxes(videopath, weightpath):
    ...
    return bbs, classes, remove_vid
```

#### Parameters

- `videopath`: The path to the input video file.
- `weightpath`: The path to the YOLOv5 model weights for object detection.

#### Returns

- `bbs`: Extracted bounding box information.
- `classes`: All class names that the model can detect.
- `remove_vid`: Flag for removing videos of low data quality.

#### Example Usage

```python
video_path = "input_video.mp4"
weightpath = "hands.pt"

bbs, classes, remove_vid = GetBoundingBoxes(videopath, weightpath)
```

#### Interpolation

After extracting the bounding box information, the script performs simple linear interpolation to fill in missing information for frames where the object was not detected and removing videos that do not fulfill the data quality requirements, e.g. has too many missing frames.


### <em>Target Centering</em>

The script inlcudes a video centering functionality which allows you to create a new video where a target object is always centered. This is useful because it effectively filters out the head movement from the videos, leaving only the true hand trajectory. The provided code takes a video file as input and applies centering transformations based on bounding box information.

```python
def Center(videopath, bounding_boxes):
    ...
    return offsets, start_cords, video_centered
```

#### Parameters

- `videopath`: The path to the input video file.
- `bounding_boxes`: A list of bounding boxes representing the target object in each frame of the video. Each bounding box should be a tuple of four values: `(x_min, y_min, x_max, y_max)`.

#### Returns

- `offsets`: Values of new video dimensions - old video dimensions.
- `start_cords`: Start coordinates tuple `(start_row, start_col)` of the centered video embedded into new video.
- `video_centered`: The target-centered video (with black borders).

#### Example Usage

```python
video_path = "input_video.mp4"
bounding_boxes = [(100, 100, 300, 300), (150, 150, 350, 350), ...]  # Bounding boxes for each frame

offsets, start_cords, video_centered = Center(video_path, bounding_boxes)
```


### <em>Visualization</em>

There are two additional functions `Export(video, targetBBs, handBBs, labelpath)` and `PlotTrajectory(path, name, input)`. The former enables saving of the labeled video with bounding boxes around hand and target object, and a vector indicating the next movement direction of the hand. The latter allows saving trajectory plots of single videos indicating the true movement trajectory of the hand towards the target.


## 2. Training

The transformer_training.py script is dedicated to training the transformer model. The model solves a classification task of predicting an angle in [0, 360] (1°) that is the direction from the current hand position to the next hand position. 

### <em>Model architecture</em>

The model architecture follows the original transformer model from ["Attention is all you need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017) stacking encoder and decoder modules that incorporate self attention layers. Its implementation is largely guided by following the TensorFlow tutorial [Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer). 

![Transformer](https://www.tensorflow.org/images/tutorials/transformer/transformer.png)

However, our solution uses data-specific embeddings, i.e. no additional context embedding as bounding box data is inherently spatial data already (and 8D), and stacked sine-embeddings for our input data as it is of circular nature. Furthermore, we use a custom padding mask for masking out padding tokens during the training. Finally, the transformer outputs logits for each angle class that are fed trough a softmax layer for further use in the prediction script.

### <em>Model Training</em>

- Deserialize the training dataset from a zipped file (`train_ds.zip`) using the TFRecord dataset.
- Build a Transformer model with specified hyperparameters.
- Compile the model using the Adam optimizer and Mean Squared Error loss function.

#### Training Loop

- Initialize logging and metrics for training and testing.
- Execute the main training loop with epochs.
  - Train the model using the `train_step` method, updating the training loss metric.
  - Test the model using the `predict` method, calculating and updating the testing loss metric.
  - Log metrics using TensorBoard.
  - Check for model checkpointing and early stopping.
- Save the trained model.

#### Example Usage

```python
# Define hyperparameters
batch_size, max_sequence_length, num_layers, num_heads, dff, input_dim, output_dim, d_model = ...

# Instantiate the Transformer model
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)

# Build the model using first batch
for con, x, target in train_ds.take(1):
    break
output = transformer((con, x))

# Compile the model
transformer.compile(
    loss=masked_loss,
    optimizer="adam",
    metrics=[masked_accuracy])

# Pseudo-function instead of train-test loop
train_transformer(...)
```


## 3. Prediction

The predictions.ipynb notebook focuses on making predictions using the trained transformer model.

### <em>Generate Predictions</em>
- Deserialize the test dataset from a zipped file (`test_ds.zip`) using the TFRecord dataset.
- Load the pre-trained Transformer model using TensorFlow's `load_model` function.
- Iterate through the DS: compute the padding mask, call the transformer and extract the highest probability prediction.

#### Example Usage

```python
import tensorflow as tf

# Load tensorflow dataset
test_ds_path = "./data/test_ds.zip"
test_dataset = tf.data.TFRecordDataset(test_ds_path, compression_type='GZIP')
test_dataset = test_dataset.map(deserialize)

# Import trained model
transformer = tf.keras.models.load_model("transformer")

# Obtain predictions
for batch, (context, x, target) in enumerate(test_dataset):
        mask = compute_mask(target, padding_token=PAD)
        logits = transformer((context, x), training=False, mask=mask)
        predictions = tf.argmax(logits, axis=2)

        print(f'{"Ground Truth"}: {target[1].numpy().flatten().tolist()}')
        print(f'{"Prediction"}: {predictions[1].numpy().flatten().tolist()}')
```