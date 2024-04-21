import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

# Load the dataset
data = pd.read_csv('student_data.csv')

# Perform one-hot encoding for the 'gender' column
data_encoded = pd.get_dummies(data, columns=['gender'])

# Split the data into features and target variable
X = data_encoded.drop('grade', axis=1)
y = data_encoded['grade']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for grid search
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}

# Perform grid search with cross-validation
scoring = {'accuracy': 'accuracy', 'f1_macro': make_scorer(f1_score, average='macro')}
stratified_cv = StratifiedKFold(n_splits=2)
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=stratified_cv, scoring=scoring, refit='f1_macro')
grid_search.fit(X_train, y_train)

# Obtain the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Perform cross-validation
scores = cross_val_score(best_model, X, y, cv=stratified_cv, scoring='accuracy')

# Calculate the average accuracy
average_accuracy = scores.mean()

# Print the evaluation metrics and average accuracy
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Average Accuracy:", average_accuracy)