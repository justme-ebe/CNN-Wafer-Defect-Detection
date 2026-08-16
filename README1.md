# Semiconductor Wafer Defect Dataset Exploration

## Overview
This repository phase contains the initial data exploration and visualization pipeline for the WM-811K semiconductor wafer defect dataset. The objective is to analyze the spatial signatures of varying defect categories prior to implementing a Convolutional Neural Network (CNN) for automated classification.

## Dataset Access
The raw dataset consists of 811,457 wafer entries. Due to file size constraints, the `.pkl` file is excluded from this repository. 
* Access the raw dataset here: [Kaggle: Defect Detection in Wafer Bin Maps](https://www.kaggle.com/code/paulbassaler/defect-detection-in-wafer-bin-maps/data)

## Data Exploration
The `data_exploration.py` script unpacks the dataset to assess class distributions and visualize spatial defect patterns. The dataset features highly imbalanced classes and variable-sized wafer maps across nine distinct failure types (e.g., Center, Donut, Edge-Ring, Scratch, etc.).

## Initial Visualizations

![Wafer Map Samples](sample_wafers.png)
*(Note: Replace `sample_wafers.png` with the exact filename of the uploaded screenshot showing the visual grid of wafer maps).*

![Dataset Statistics](dataset_stats.png)
*(Note: Replace `dataset_stats.png` with the exact filename of the uploaded screenshot showing the training and test dataset counts).*