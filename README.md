# Study Project: Sensory augmentation and grasping movements

This repository contains the study project on Sensory augmentation and grasping movements in SS23 & WS23/24. The project builds on the OptiVisT project, which focuses on supporting blind individuals in guiding their hand movements using a tactile bracelet and vibration motor inputs. In this study, we aim to enhance this concept by implementing a transformer neural network for hand movement guidance. Instead of simple left, right, up, and down commands from the bracelet, we translate naturalistic movement vectors into corresponding directional vibrations on an improved version of the tactile bracelet.

## Dependencies

The code relies on the following external libraries:

- **cv2**: OpenCV library for video processing.
- **numpy**: Numerical computing library for efficient array manipulation.
- **ultralytics**: Ultralytics library for object detection (used to obtain bounding box information).
- **pydrive**: Library for interfacing with Google Drive.
- **tensorflow**: An open-source machine learning framework.
- **keras**: High-level neural networks API running on top of TensorFlow.
- **Progbar**: Progress bar for tracking training iterations.
- **TensorBoard**: TensorFlow's built-in tool for visualizing training metrics.

Please ensure that these libraries are installed and available before running the code.

## Notebooks

### Video Centering

The 01-transformer-video_centering.ipynb notebook inlcudes a video centering functionality which allows you to create a new video where a target object is always centered. This is useful for ensuring that a specific object remains in the center of each frame throughout the video. The provided code takes a video file as input and applies centering transformations based on bounding box information.

#### Function Signature

```python
def Center(videopath, bounding_boxes, width, height, to_path):
    ...
```

#### Parameters

- `videopath`: The path to the input video file.
- `bounding_boxes`: A list of bounding boxes representing the target object in each frame of the video. Each bounding box should be a tuple of four values: `(x_min, y_min, x_max, y_max)`.
- `width`: The desired width of the output video.
- `height`: The desired height of the output video.
- `to_path`: The path where the centered video should be saved.

#### Example Usage

```python
video_path = "input_video.mp4"
bounding_boxes = [(100, 100, 300, 300), (150, 150, 350, 350), ...]  # Bounding boxes for each frame
width = 640
height = 480
output_path = "centered_video.mp4"

Center(video_path, bounding_boxes, width, height, output_path)
```

The resulting video will have the target object consistently positioned at the center of each frame. Said videos are then saved, and the final bounding box information for the target and hand (post-centering) is stored in NPZ files and uploaded to Google Drive.

### Data Creation
The 02-transformer-data_creation.ipynb notebook focuses on generating and preprocessing the training data required for training the transformer neural network. It involves extracting bounding box information from videos, performing interpolation to fill missing information, and concatenating the input data for each video.

#### Function: `GetBoundingBoxes(videopath)`

This function extracts bounding boxes from a video using an object tracker. It takes the video's path as input and returns a tuple containing bounding box information (`bbs`) and class names (`classes`).

#### Function Signature

```python
def GetBoundingBoxes(videopath):
    ...
```

#### Parameters

- `videopath`: The path to the input video file.

#### Video Processing Workflow

The script starts by loading a YOLO model (`yolov8n.pt`) and setting video folder paths. It iterates through each video in the specified folder, loading post-centering bounding boxes for hand and target objects. The script then creates an array (`vid_input`) with information for each frame, performing interpolation for both target and hand objects. The interpolated data is used to fill the `vid_input` array, which is then concatenated together (`input`). Rows with NaN values are removed for data cleanliness.

#### Interpolation
After extracting the bounding box information, the notebook performs interpolation to fill in missing information for frames where the object was not detected. The interpolation method currently used takes the last known information for the object.

### Training
The 03-transformer-training.ipynb notebook is dedicated to training the transformer model. 

#### Import Dataset
- Deserialize the training dataset from a zipped file (`train_ds.zip`) using the TFRecord dataset.

#### Build Model
- Build a Transformer model with specified hyperparameters.
- Compile the model using the Adam optimizer and Mean Squared Error loss function.

#### Train Model
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

# Instantiate and train the Transformer model
transformer = Transformer(num_layers, d_model, num_heads, dff, max_sequence_length, input_dim, output_dim)
train_model(transformer, train_ds, test_ds, epochs=100, lr=0.001, patience=10)
```

### Prediction
The 04-transformer-prediction.ipynb notebook focuses on making predictions using the trained transformer model.

#### Import Trained Model
- Load the pre-trained Transformer model using TensorFlow's `load_model` function.

#### Generate Predictions
- Generate dummy novel data with a specified input shape based on the training data.
- Use the loaded model to obtain predictions for the input data.

#### Example Usage
```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import trained model
transformer = load_model("transformer")

# Generate predictions
batches = 2
batch_size = 3
sequence_length = 126
input_dim = 8
input_data = tf.random.uniform((batch_size, sequence_length, input_dim))

# Obtain predictions
predictions = transformer.predict(input_data)
print(f"Shape: {predictions.shape}, \nPredictions: \n{predictions}")
```

