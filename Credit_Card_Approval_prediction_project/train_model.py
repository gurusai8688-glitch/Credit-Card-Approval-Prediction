import pandas as pd

# Load datasets
application = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

# Merge datasets using ID
data = application.merge(credit, on="ID", how="inner")

# Display first 5 rows
print(data.head())

# Dataset information
print("\nShape:", data.shape)
print("\nColumns:")
print(data.columns)

# Missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove rows with missing values
data = data.dropna()

print("\nAfter Cleaning:")
print(data.shape)

data = data.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(data.shape)
 # Drop unnecessary column
data = data.drop(columns=["ID"])
print("\nCleaned Data:")
print(data.head())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

text_columns = data.select_dtypes(include=['object']).columns

for col in text_columns:
    print(f"Encoding: {col}")
    data[col] = le.fit_transform(data[col].astype(str))

print("\nAfter Label Encoding:")
print(data.head())
from sklearn.model_selection import train_test_split

# Features (X) and Target (y)
X = data.drop(["STATUS"], axis=1)
y = data["STATUS"]

print(X.columns)
print(X.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "credit_card_model.pkl")

print("Model Saved Successfully!")