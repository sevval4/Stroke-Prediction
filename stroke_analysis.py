import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Fill missing values in 'bmi' with mean
dataset["bmi"].fillna(dataset["bmi"].mean(), inplace=True)

# Visualize target variable distribution and other insights
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
sns.countplot(x="stroke", data=dataset, ax=axes[0, 0])
sns.countplot(x="stroke", hue="gender", data=dataset, ax=axes[0, 1])
sns.boxplot(x="stroke", y="age", data=dataset, ax=axes[1, 0])
sns.boxplot(x="stroke", y="avg_glucose_level", data=dataset, ax=axes[1, 1])
plt.show()

# Categorical variable visualizations
def plot_categorical_dist(dataset, categorical_feature, rows, cols, plot_type):
    fig, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
    features = dataset.select_dtypes(include=["float64", "int64"]).columns.values
    counter = 0

    for i in range(rows):
        for j in range(cols):
            if counter < len(features):
                feature = features[counter]
                sns_plot_func = getattr(sns, plot_type)
                sns_plot_func(x=categorical_feature, y=feature, data=dataset, ax=axarr[i, j])
                axarr[i, j].set_title(f"{feature} vs {categorical_feature}")
                counter += 1
            else:
                axarr[i, j].axis("off")

    plt.tight_layout()
    plt.show()

plot_categorical_dist(dataset=dataset, categorical_feature="stroke", rows=3, cols=4, plot_type="stripplot")

# Convert categorical features to numeric with one-hot encoding
dataset = pd.get_dummies(dataset, drop_first=True)

# Correlation heatmap
corr_matrix = dataset.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Prepare data for model training
X = dataset.drop("stroke", axis=1)
y = dataset["stroke"]

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Model evaluation
models = [
    ("LR", LogisticRegression()),
    ("NB", GaussianNB()),
    ("KNN", KNeighborsClassifier()),
    ("DT", DecisionTreeClassifier()),
    ("SVM", SVC()),
    ("LDA", LinearDiscriminantAnalysis())
]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"== {name} ==")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=1)}")
