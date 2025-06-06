# Breast cancer diagnosis predictor

## Overview

The Breast Cancer Diagnosis app is a machine learning-powered tool designed to assist medical professionals in diagnosing breast cancer. Using a set of measurements, the app predicts whether a breast mass is benign or malignant.

Public dataset [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) was used.

A live version of the application can be found on [Streamlit Community Cloud](https://app-cancer-predict-sv3lzycxo8pvf43e8q2497.streamlit.app/). 


## Installation

You can run this inside a virtual environment to make it easier to manage dependencies. I recommend using `conda` to create a new environment and install the required packages. You can create a new environment called `breast-cancer-diagnosis` by running:

```bash
conda create -n breast-cancer-diagnosis python=3.10 
```

Then, activate the environment:

```bash
conda activate breast-cancer-diagnosis
```

Then, activate the environment:

```bash
conda activate breast-cancer-diagnosis
```

To install the required packages, run:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies, including Streamlit, OpenCV, and scikit-image.

## Usage
To start the app, simply run the following command:

```bash
streamlit run app/main.py
```


