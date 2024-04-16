import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000

# Features: chemical composition
chemicals = ['A', 'B', 'C', 'D', 'E']
data = pd.DataFrame(np.random.choice(chemicals, size=(num_samples, len(chemicals))), columns=chemicals)

# Target variable: recyclability (binary: 0 - non-recyclable, 1 - recyclable)
data['recyclability'] = np.random.randint(0, 2, size=num_samples)

# Save the dataset to a CSV file
data.to_csv('chemical_recycling_dataset.csv', index=False)

# Step 2: Load and preprocess the data
data = pd.read_csv('chemical_recycling_dataset.csv')
X = data.drop('recyclability', axis=1)
y = data['recyclability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Encode categorical variables
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Step 5: Build and train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)