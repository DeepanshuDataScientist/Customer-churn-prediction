# Customer Churn Prediction

## Overview
This project aims to predict customer churn using machine learning techniques. Customer churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company. Predicting churn can be crucial for businesses to retain customers and optimize their strategies. 

## Data
The dataset used in this project is Telcom-Customer-Churn which contains information about Telcom customers including their tenure, monthly charges, total charges, and whether they churned or not.

## Preprocessing
- Removed the 'customerID' column as it is not relevant for modeling.
- Handled missing values in the 'TotalCharges' column by converting them to numeric values and replacing them with NaN.
- Converted categorical variables to numerical format.
- Scaled numerical features using MinMaxScaler.
- Split the data into training and testing sets.

## Model
Built a neural network model using TensorFlow and Keras to predict customer churn:
- Input layer with 26 neurons (number of features)
- Two hidden layers with 20 and 15 neurons respectively, using ReLU activation function
- Output layer with 1 neuron and sigmoid activation function for binary classification

Compiled the model with Adam optimizer and binary cross-entropy loss function.

## Training
Trained the model on the training data with 100 epochs.

## Evaluation
Evaluated the model on the testing data and analyzed performance using metrics such as accuracy, precision, recall, and confusion matrix.

## Results
The trained model achieved an accuracy of approximately 80%. Precision, recall, and confusion matrix were used to evaluate the performance of the model.

## Visualization
Visualized the distribution of features related to tenure, monthly charges, and churn to gain insights into the data.

## Libraries Used
- Pandas
- Matplotlib
- NumPy
- TensorFlow
- Scikit-learn
- Seaborn

## Instructions
1. Ensure all necessary libraries are installed (listed in requirements.txt).
2. Download the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Run the provided code to preprocess the data and train the model.
4. Evaluate the model performance and analyze the results.
5. Make predictions on new data using the trained model.

For detailed implementation and code, refer to the provided script.
