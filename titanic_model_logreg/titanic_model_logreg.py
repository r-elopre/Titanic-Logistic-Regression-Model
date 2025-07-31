import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Load and Inspect Data
print("Loading and inspecting the dataset...")
data_path = r'C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\data\train_scaled.csv'

df = pd.read_csv(data_path)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check class balance in target variable
print("\nSurvived distribution:")
print(df['Survived'].value_counts(normalize=True))

# Step 2: Prepare Features and Target
# Exclude PassengerId and Survived from features
X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived']
print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Step 3: Train-Test Split
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Step 4: Train Logistic Regression Model
print("\nTraining logistic regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# Step 5: Make Predictions
y_pred = model.predict(X_test)
print("\nFirst 5 test predictions:", y_pred[:5])

# Step 6: Evaluate Model
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 7: Plot Confusion Matrix
print("\nGenerating confusion matrix plot...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# Step 8: Save Predictions
# Create DataFrame with PassengerId and predictions
predictions_df = pd.DataFrame({
    'PassengerId': df.loc[X_test.index, 'PassengerId'],
    'Survived': y_pred
})
predictions_df.to_csv(
    r'C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\data\titanic_logreg_predictions.csv',
    index=False
)

print("\nPredictions saved to 'titanic_logreg_predictions.csv'")

# Step 9: Summary
print("\nSummary:")
print("The logistic regression model was trained on the Titanic dataset.")
print("Key steps: data loading, train-test split, model training, evaluation, and visualization.")
print("The model serves as a baseline for predicting passenger survival.")