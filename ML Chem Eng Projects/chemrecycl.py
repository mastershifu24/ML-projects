##Data Preparation: The code begins by loading a dataset containing information about chemical substances and their recyclability. The dataset is divided into two parts: one containing the features (characteristics) of the substances and another containing the corresponding recyclability labels.

##Splitting the Data: The dataset is split into two sets—training and testing sets. The training set is used to teach the machine learning model, while the testing set is used to evaluate its performance.

##Data Encoding and Scaling: To ensure the data is in a suitable format for the machine learning model, any categorical variables in the dataset are converted into numerical representations through a process called encoding. Additionally, the numerical features are scaled to a standardized range to prevent any particular feature from dominating the model's learning process.

##Training the Model: A logistic regression model is chosen and trained using the training set. Logistic regression is a type of machine learning algorithm commonly used for classification tasks, such as predicting whether a chemical substance is recyclable or not. The model learns patterns and relationships within the training data to make predictions.

##Making Predictions: Once the model is trained, it is used to make predictions on the testing set. It takes the features of the chemical substances from the testing set and predicts their recyclability based on what it has learned during training.

##Evaluating the Model: The accuracy of the model's predictions is calculated by comparing them with the true recyclability labels from the testing set. This metric indicates the percentage of correct predictions made by the model.

##Understanding Model Performance: The code provides additional insights into the model's performance. The classification report summarizes metrics such as precision, recall, and F1-score, which help assess the model's performance for each class (recyclable or non-recyclable). The confusion matrix displays the count of correct and incorrect predictions, providing a deeper understanding of the model's performance on different classes.

##Feature Importance: The code also examines the importance of each feature in the model's decision-making process. It calculates the absolute value of the coefficients assigned to each feature by the logistic regression model. The higher the coefficient's absolute value, the more influential the corresponding feature is in predicting recyclability.

##By running this code, we can gain insights into the performance and interpretability of the machine learning model, enabling us to understand how well it predicts the recyclability of chemical substances and which features contribute most to its predictions.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
data = pd.read_csv('chemical_recycling_dataset.csv')
X = data.drop('recyclability', axis=1)
y = data['recyclability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical variables
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Obtain feature importance
feature_importance = pd.DataFrame({'Feature': X_train_encoded.columns, 'Importance': np.abs(model.coef_[0])})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)
