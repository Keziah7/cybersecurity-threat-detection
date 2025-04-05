import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Start message
print("Cybersecurity Threat Detection Project Started!")

# Load the dataset
file_name = "network_data.csv"  # Update with the correct filename if needed
try:
    data = pd.read_csv(file_name)
    print("\nDataset Loaded Successfully!")
except FileNotFoundError:
    print(f"Error: {file_name} not found! Please check the file path.")
    exit()

# Display basic dataset info
print("\nDataset Shape:", data.shape)
print("\nFirst 5 Rows:\n", data.head())

# Display column names
print("\nColumn Names:\n", data.columns)

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values in Each Column:\n", missing_values[missing_values > 0])

# Drop rows with missing values
data = data.dropna()
print("\nMissing values removed.")

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_columns)

# Convert categorical columns to numerical using Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for future reference

print("\nCategorical data converted to numerical format.")

# Normalize numerical columns
scaler = StandardScaler()
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

print("\nNumerical data normalized.")

# Save the processed data
processed_file = "processed_network_data.csv"
data.to_csv(processed_file, index=False)
print(f"\nPreprocessed data saved as '{processed_file}'.")

# Reload the processed data
data = pd.read_csv(processed_file)

# Identify the attack label column
possible_labels = ["label", "attack_cat", "class", "normal"]  # Common attack label names
attack_col = None

for col in data.columns:
    if col.lower() in possible_labels:
        attack_col = col
        break

if attack_col:
    print(f"\nDetected attack label column: {attack_col}")

    # Step 1: Check unique values in the attack column
    print("\nUnique values in the attack label column:\n", data[attack_col].unique())

    # Convert the attack labels to categorical if needed
    data[attack_col] = data[attack_col].astype('category')
    y = data[attack_col].cat.codes  # Encode categorical labels as numeric values

    # Plot attack type distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x=data[attack_col], hue=data[attack_col], palette='coolwarm', legend=False)
    plt.title("Distribution of Attack Types")
    plt.xticks(rotation=45)

    # Save the figure
    plt.savefig("attack_distribution.png")
    plt.show()
    print("Saved attack distribution graph as 'attack_distribution.png'.")
else:
    print("\nWarning: No attack label column found! Please verify your dataset.")

# Plot correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")

# Save the figure
plt.savefig("feature_correlation.png")
plt.show()
print("Saved feature correlation graph as 'feature_correlation.png'.")

# Step 2: Separate features and target
if attack_col:
    X = data.drop(columns=[attack_col])
else:
    print("Attack label column not found. Exiting...")
    exit()

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
print("\n🎯 Model Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
