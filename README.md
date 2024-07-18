
# CNN-Based Potato Disease Detection

## Overview

This repository contains a deep learning model that uses Convolutional Neural Networks (CNN) to classify potato leaf images into three categories: Healthy, Early Blight, and Late Blight. The goal is to aid farmers in the early detection of diseases in their potato crops, enabling timely intervention and treatment.

## Features

- **Model Architecture**: Utilizes a CNN model built with TensorFlow and Keras.
- **Data Preprocessing**: Includes scripts for preprocessing the dataset to prepare it for training.
- **Training and Evaluation**: Detailed instructions and scripts for training the model and evaluating its performance.
- **Deployment**: FastAPI backend for real-time prediction and integration with other applications.

## Dataset

The dataset used for this project is a collection of potato leaf images labeled as Healthy, Early Blight, or Late Blight. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/adhityaram18/CNN-Based-Potato-Disease-Detection.git
    cd CNN-Based-Potato-Disease-Detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and prepare the dataset:
    ```bash
    # Follow the instructions in the Dataset section to download the dataset.
    ```

## Usage

### Training the Model

1. Preprocess the dataset:
    ```bash
    python preprocess_data.py
    ```

2. Train the model:
    ```bash
    python train_model.py
    ```

### Running Predictions

1. Start the FastAPI server:
    ```bash
    uvicorn api:app --reload
    ```

2. Make predictions by sending a POST request to the `/predict` endpoint with an image file.

## Results

The model achieves an accuracy of over 90% in classifying the potato leaf images into the three categories. Detailed metrics and visualizations of the model's performance are available in the evaluation scripts.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The dataset is sourced from the [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
- Special thanks to the contributors and maintainers of TensorFlow, Keras, and FastAPI for their excellent libraries and documentation.

