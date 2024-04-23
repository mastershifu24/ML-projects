# Import the required libraries
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Sample dataset with features and corresponding disease labels
# Assume each data point has three features (e.g., age, blood pressure, cholesterol level)
# and the corresponding labels represent whether the individual has a disease (1) or not (0)
features = [[40, 120, 200], [60, 130, 220], [45, 115, 190], [50, 125, 210], [55, 135, 230]]
labels = [0, 1, 0, 1, 1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create an SVM classifier
clf = svm.SVC(kernel='linear')

# Define the hyperparameter grid for tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=2)
grid_search.fit(X_train, y_train)

# Get the best SVM model from grid search
best_clf = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)