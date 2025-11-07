# ML Classification Web App

A simple browser-based machine learning application that performs classification using three different algorithms.

## Features

- **CSV File Upload**: Upload your dataset directly in the browser
- **Data Cleaning**: Automatic handling of missing values using:
  - Linear interpolation for numeric columns
  - Mode imputation for categorical columns
- **Three ML Models**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Naive Bayes
- **Performance Comparison**: Visual chart comparing model accuracies
- **No Server Required**: Runs entirely in the browser

## Usage

1. Open `index.html` in a web browser
2. Click "Choose CSV File" and select your dataset
3. The last column will be used as the target variable
4. Click "Train Models" to train and evaluate all three models
5. View the accuracy results and comparison chart


## Technologies Used

- HTML5
- CSS3
- JavaScript (ES6+)
- Papa Parse (CSV parsing)
- ml.js libraries (Machine Learning)

