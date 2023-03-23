# Wildfire Prediction
This repository contains Jupyter notebooks for the analysis of a wildfire dataset, data collection from Mapbox API, and the training of a Convolutional Neural Network (CNN) to predict whether an area is at risk of a wildfire or not.

## Dataset
The original dataset used in this project is Canada's Wildfires dataset, which contains information about wildfire occurrences in Canada from 1918 to 2020. The dataset includes information about the location, size, and cause of the wildfires. The dataset is available in CSV format and contains coordinates for each wildfire.

## Notebooks
1. data_notebook.ipynb: This notebook contains code to collect images from Mapbox API using the coordinates from the wildfire dataset. The images are saved locally for later use and split into training and validation sets.

2. WildfireAnalysis.ipynb: This notebook contains exploratory data analysis (EDA) of the wildfire dataset. It includes visualizations of the wildfire occurrences and their causes.

3. wildfire_prediction.ipynb: This notebook contains code to prepare the collected images for training the CNN. The images are resized, normalized and contains code to train a CNN using the prepared image dataset. The CNN is trained to predict whether an area is at risk of a wildfire or not.

## Requirements
- Python 3.x
- Jupyter Notebook
- TensorFlow
- Matplotlib
- NumPy
- Pandas
- Mapbox API Key

## New Collected Dataset
https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset


## Credits
The original wildfire dataset used in this project is from the Government of Canada's Open Data Portal.