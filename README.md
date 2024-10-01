Stroke Prediction Data Analysis and Classification Models


Project Overview
This project focuses on the analysis and prediction of stroke occurrence using a healthcare dataset. The goal is to build, evaluate, and compare various machine learning models to predict the likelihood of a patient having a stroke based on various health indicators.

Dataset
The dataset used in this project is the Healthcare Dataset Stroke Data. It contains various attributes such as age, gender, BMI, average glucose levels, and more, with the target variable being stroke (indicating whether a patient had a stroke or not). The data has been pre-processed to fill missing values and encode categorical variables.

Requirements
To run this project, you need the following Python libraries:
pandas
matplotlib
seaborn
scikit-learn
mglearn
yellowbrick
numpy

You can install these dependencies using:
pip install -r requirements.txt

Exploratory Data Analysis (EDA)
The project starts with an exploratory data analysis (EDA) phase to understand the distribution of the data and relationships between features and the target variable (stroke). Various visualizations such as count plots, box plots, and correlation matrices have been generated to explore the data.

Missing values in the bmi column were filled with the mean.
Visualizations were created to show the distribution of stroke occurrences based on factors like gender, age, and average glucose level.
Model Training and Evaluation
Six machine learning models were trained and evaluated for stroke prediction:

Logistic Regression
Gaussian Naive Bayes
K-Nearest Neighbors
Decision Tree Classifier
Support Vector Machine
Linear Discriminant Analysis
Each model was evaluated using accuracy, confusion matrix, and classification report metrics. Additionally, cross-validation was performed to measure the robustness of the models.

Results
The models were evaluated and their results, including accuracy and classification metrics, were compared to determine the best-performing model for stroke prediction.
