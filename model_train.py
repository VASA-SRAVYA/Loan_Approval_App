# model_train.py

import pappndas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('loan_data_large.csv')

# Drop Loan_ID
df.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical columns
label_cols = ['Gender', 'Married', 'Dependents', 'Education', 
              'Self_Employed', 'Property_Area', 'Loan_Status']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Feature-Target split
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('loan_model.pkl', 'wb'))
print("✅ Model saved as loan_model.pkl")
