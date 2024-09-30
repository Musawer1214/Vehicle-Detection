# Vehicle Detection using YOLOv5

This repository contains a project that uses a YOLOv5 model to detect multiple types of vehicles in images and videos. The trained model can detect various types of vehicles such as cars, trucks, and motorcycles, making it useful for real-time traffic monitoring, autonomous driving, and surveillance systems.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Usage](#usage)
  - [Installation](#installation)
  - [Training the Model](#training-the-model)
  - [Resume Training](#resume-training)
  - [Testing the Model](#testing-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project utilizes the YOLOv5 model to detect multiple types of vehicles in images and video streams. YOLOv5 is a fast, highly accurate object detection model designed for real-time applications. The model is capable of detecting vehicles with high precision, making it suitable for a variety of applications in traffic analysis and autonomous systems.

## Model Architecture
YOLOv5 is an advanced deep learning architecture that builds on the YOLO (You Only Look Once) family of models, known for their speed and accuracy in object detection tasks. YOLOv5 can be easily trained on custom datasets, which makes it ideal for detecting vehicles in different environments.

## Dataset
The model was trained on a custom dataset of vehicle images that includes various vehicle types such as:
- Cars
- Trucks
- Motorcycles
- Buses

The dataset is split into training, validation, and test sets for proper evaluation of the model's performance.

## Training

The YOLOv5 model was trained using the following parameters:
- **Batch size**: 64
- **Image size**: 640x640
- **Epochs**: 50

### Model Performance
The model was evaluated on a test set of vehicle images and videos and achieved a high detection rate, demonstrating its robustness in real-world conditions.

## Usage

### Installation
Before running the model, you need to install the necessary libraries. You can run the project on platforms like Jupyter, Kaggle, or Google Colab.

#### Install Necessary Libraries
```bash
%pip install ultralytics
import ultralytics
ultralytics.checks()
```

### Training the Model
Train the YOLOv5 model on your dataset by using the following command:

```bash
# Train YOLOv8n on COCO8 for 50 epochs
!yolo train model= yolov8n.pt data="/kaggle/input/gun-dataset/data.yaml" epochs=10 imgsz=640 batch=64
```

### Resume Training
To resume training from a previous checkpoint, use this command:

```bash
# Resume training
!yolo detect train data= "/kaggle/input/new-weopon-detection/data.yaml" model="/kaggle/working/runs/detect/train6/weights/best.pt" resume=True
```

### Testing the Model
To test the trained model on an image or video, run the following command:

```bash
# Run inference on an image or video with YOLOv8n
!yolo predict model="/kaggle/input/weapon-detection/pytorch/default/1/best (11).pt" source='/kaggle/input/gun-dataset/DARRA ADAM KHEL VLOGS.mp4'
```

### Platforms
These commands can be executed on:
- Jupyter Notebooks
- Kaggle Kernels
- Google Colab

## Results
The YOLOv5 model successfully detects various types of vehicles in both images and video streams. The output includes bounding boxes around detected vehicles, along with labels for each detected object.

## Contributing
Contributions are welcome! Feel free to submit a Pull Request or open an issue for any improvements or suggestions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

This `README.md` file includes the details you provided and explains how to use YOLOv5 for vehicle detection. The commands and instructions are adaptable to Jupyter, Kaggle, or Google Colab environments, as specified.
