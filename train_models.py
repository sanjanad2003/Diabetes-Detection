import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Ensure the models directory exists
models_dir = './models_dir/'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load dataset
df = pd.read_csv('diabetes.csv')

# Split data
X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)

# Save models
with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'wb') as f:
    pickle.dump(log_reg, f)

with open(os.path.join(models_dir, 'decision_tree.pkl'), 'wb') as f:
    pickle.dump(dec_tree, f)

with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Save metrics
metrics = {
    "logistic_regression": classification_report(y_test, log_reg.predict(X_test), output_dict=True),
    "decision_tree": classification_report(y_test, dec_tree.predict(X_test), output_dict=True)
}

with open(os.path.join(models_dir, 'metrics.pkl'), 'wb') as f:
    pickle.dump(metrics, f)

print("✅ Models trained and saved successfully!")
