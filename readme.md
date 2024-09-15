# Multimodal Emotion Recognition Using Audio and Video Fusion

This project implements a deep learning model for multimodal emotion recognition by combining audio and video features. The model uses audio features extracted from Mel-frequency cepstral coefficients (MFCCs) and video features extracted from frames using 3D convolutional neural networks (3D CNNs). This approach allows the model to learn from both audio and visual cues to accurately predict emotions.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Feature Extraction](#feature-extraction)
    - [Audio Feature Extraction](#audio-feature-extraction)
    - [Video Feature Extraction](#video-feature-extraction)
4. [Model Architecture](#model-architecture)
    - [Audio Model](#audio-model)
    - [Video Model](#video-model)
    - [Fusion](#fusion)
5. [Training](#training)
6. [Inference](#inference)
7. [How to Run](#how-to-run)
8. [Dependencies](#dependencies)
9. [Git Setup](#git-setup)

## Introduction

Emotion recognition is an important task in human-computer interaction, where the goal is to detect and interpret emotions from various inputs, such as speech and facial expressions. This project implements a multimodal approach, fusing audio and video data to improve the accuracy of emotion recognition.

## Dataset

The dataset used in this project is a subset of The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), which can be accessed [here](https://zenodo.org/records/1188976). This dataset includes 24 professional actors (12 male, 12 female) vocalizing two lexically matched statements in a neutral North American accent. The subset used includes audio and video recordings of actors expressing various emotions.

### Dataset Details:

- **Modality:** Audio-visual
- **Emotions:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **Format:** `.wav` for audio, `.mp4` for video
- **Organization:** Data is organized by actor, with each file labeled according to the emotion expressed.

### File Structure:

```
dataset/
│
├── audios/
│   ├── Actor_01/
│   │   ├── 03-02-05-01-02-01-01.wav
│   │   └── ...
│   ├── Actor_02/
│   │   ├── 03-02-05-01-02-01-02.wav
│   │   └── ...
│   └── ...
│
└── videos/
    ├── Actor_01/
    │   ├── 03-02-05-01-02-01-01.mp4
    │   └── ...
    ├── Actor_02/
    │   ├── 03-02-05-01-02-01-02.mp4
    │   └── ...
    └── ...
```

## Feature Extraction

### Audio Feature Extraction

Audio features are extracted using Mel-frequency cepstral coefficients (MFCCs), which are commonly used in speech and audio processing.

**Steps:**
1. Load the audio file using `librosa`.
2. Apply random augmentations, such as pitch shift, time stretch, and noise addition, to make the model more robust.
3. Extract MFCCs with 40 coefficients over time.
4. Adjust the time steps to a fixed length (e.g., 44) by truncating or padding with zeros.

### Video Feature Extraction

Video features are extracted using a sequence of frames processed by a 3D CNN model.

**Steps:**
1. Read video frames using `cv2.VideoCapture`.
2. Resize each frame to 112x112 pixels.
3. Adjust the number of frames to a fixed sequence length (e.g., 20) by truncating or padding with black frames.
4. Reshape frames to match the input shape required by the model: `(batch_size, sequence_length, height, width, channels)`.

## Model Architecture

### Audio Model

The audio model processes the MFCC features through 2D convolutional layers:

- Input shape: `(40, 44, 1)` (40 MFCC coefficients, 44 time steps, 1 channel).
- 2D convolutional layers with batch normalization and dropout for regularization.
- Dense layers to learn audio representations.

### Video Model

The video model uses a 3D CNN to process the sequence of video frames:

- Input shape: `(sequence_length, 112, 112, 3)`.
- 3D convolutional layers with batch normalization and max pooling.
- Dense layers to learn video representations.

### Fusion

The outputs of the audio and video models are concatenated to form a combined feature vector, which is passed through additional dense layers to produce the final emotion prediction.

## Training

The model is trained using the combined audio and video data with the following settings:

- Loss function: Sparse categorical cross-entropy.
- Optimizer: Adam with a learning rate scheduler and gradient clipping.
- Callbacks: Early stopping, model checkpointing, and learning rate reduction on plateau.
- Data generators handle batching and on-the-fly data augmentation.

## Inference

The trained model can be used for inference on new audio and video data:

1. Load the saved model.
2. Extract features from a new audio file and a video.
3. Predict the emotion and display the results with bounding boxes and labels on the video frames.

## How to Run

1. **Prepare the Dataset:**
   - Place audio and video files in the appropriate folders under `dataset/audios` and `dataset/videos`.

2. **Install Dependencies:**
   - Use the provided `requirements.txt` to install necessary packages:
     ```
     pip install -r requirements.txt
     ```

3. **Train the Model:**
   - Run the training script:
     ```bash
     python train_model.py
     ```

4. **Inference:**
   - Run the inference script with a new video file:
     ```bash
     python run_inference.py --video_path path/to/video.mp4
     ```

## Dependencies

- Python 3.7+
- TensorFlow 2.x
- Keras
- Librosa
- OpenCV
- NumPy
- Scikit-learn

Ensure that these dependencies are installed for the code to run successfully.

