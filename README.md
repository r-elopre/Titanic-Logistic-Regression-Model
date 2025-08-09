# Titanic Logistic Regression Model
<p align="center">
  <a href="https://youtu.be/H7h5GMPuOb0?si=N8hVZrpIyc47wEfm">
    <img src="https://img.youtube.com/vi/H7h5GMPuOb0/maxresdefault.jpg" alt="Titanic Logistic Regression Model Video">
  </a>
  <br/>
  <em>Click the thumbnail to watch on YouTube</em>
</p>

## Project Overview
This project trains a logistic regression model to predict passenger survival on the Titanic using the preprocessed dataset `train_scaled.csv`. The goal is to build a baseline machine learning model for binary classification (survived or not survived), evaluate its performance, and visualize the results. The project includes data loading, train-test split, model training, evaluation with accuracy and confusion matrix, visualization using Seaborn, and saving predictions to a CSV file. This serves as a foundational exercise in machine learning, demonstrating key skills like data preprocessing, model training, and evaluation.

The project is implemented in a Python script (`titanic_model_logreg.py`) and documented in a Markdown file (`titanic_logreg_documentation.md`) for portfolio purposes.

## Dataset
The dataset (`train_scaled.csv`) contains 891 rows and 18 columns, with the following features:
- **Target**: `Survived` (0 = Not Survived, 1 = Survived)
- **Numerical Features** (standardized): `Age`, `SibSp` (siblings/spouses aboard), `Parch` (parents/children aboard), `Fare`
- **Categorical Features** (one-hot encoded): `Pclass` (passenger class), `Sex_male`, `Embarked_q`, `Embarked_s`, `Deck_B`, `Deck_C`, `Deck_D`, `Deck_E`, `Deck_F`, `Deck_G`, `Deck_T`, `Deck_Unknown`
- **Identifier**: `PassengerId` (excluded from training)

### Dataset Insights
- **Missing Values**: None detected (verified by `df.isnull().sum()`).
- **Class Distribution**: Approximately 61.6% Not Survived (0), 38.4% Survived (1), indicating moderate class imbalance.
- **Preprocessing**: Numerical features are scaled (mean ~0, std ~1), and categorical features are one-hot encoded, making the data suitable for logistic regression.

## Methodology
The Python script (`titanic_model_logreg.py`) executes the following steps:
1. **Load Libraries**: Imports `pandas`, `numpy`, `scikit-learn`, `seaborn`, and `matplotlib` for data handling, modeling, and visualization.
2. **Load and Inspect Data**: Loads `train_scaled.csv`, checks for missing values, and displays class distribution.
3. **Prepare Features and Target**: Excludes `PassengerId` and `Survived` from features (`X`), sets `Survived` as the target (`y`).
4. **Train-Test Split**: Splits data into 80% training (712 samples) and 20% testing (179 samples) with `random_state=42` for reproducibility.
5. **Train Logistic Regression**: Trains a logistic regression model with `max_iter=1000`.
6. **Make Predictions**: Generates predictions on the test set.
7. **Evaluate Model**: Computes accuracy, classification report, and confusion matrix.
8. **Visualize Confusion Matrix**: Creates a heatmap using Seaborn, saved as `confusion_matrix.png`.
9. **Save Predictions**: Saves test predictions with `PassengerId` to `titanic_logreg_predictions.csv`.
10. **Summary**: Prints a summary of the process and outcomes.

## Results
The model was trained and evaluated, producing the following results:
- **Accuracy**: 0.8212 (82.12% of test predictions were correct).
- **Classification Report**:
  ```
              precision    recall  f1-score   support
  Not Survived       0.83      0.87      0.85       105
      Survived       0.80      0.76      0.78        74
      accuracy                           0.82       179
     macro avg       0.82      0.81      0.81       179
  weighted avg       0.82      0.82      0.82       179
  ```
  - Precision: 83% for Not Survived, 80% for Survived.
  - Recall: 87% for Not Survived, 76% for Survived.
  - F1-score: 85% for Not Survived, 78% for Survived.
- **Confusion Matrix**:
  ```
  [[91 14]
   [18 56]]
  ```
  - True Negatives (Not Survived, predicted correctly): 91
  - False Positives (Not Survived, predicted as Survived): 14
  - False Negatives (Survived, predicted as Not Survived): 18
  - True Positives (Survived, predicted correctly): 56

### Confusion Matrix Visualization
Below is an ASCII representation of the confusion matrix for quick reference:
```
                 Predicted
                | Not Survived | Survived |
Actual Not Survived |     91      |    14    |
Actual Survived     |     18      |    56    |
```
A detailed heatmap is saved as `confusion_matrix.png` in the project directory.

- **Outputs**:
  - **Plot**: Confusion matrix saved as `confusion_matrix.png`.
  - **Predictions**: Test predictions saved as `titanic_logreg_predictions.csv`.

## Key Learnings
- **Train-Test Split**: Ensures model evaluation on unseen data, preventing overfitting.
- **Logistic Regression**: A robust baseline for binary classification, effective with preprocessed data.
- **Evaluation Metrics**: Accuracy provides an overall performance measure; the confusion matrix and classification report offer detailed insights into class-specific performance.
- **Visualization**: Seaborn’s heatmap and ASCII representation enhance interpretability of the confusion matrix.
- **Baseline Models**: Logistic regression serves as a benchmark for comparing more complex models.

## Files
- **Script**: `titanic_model_logreg.py` (main Python script, located at `C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\titanic_model_logreg\titanic_model_logreg.py`)
- **Dataset**: `train_scaled.csv` (preprocessed Titanic dataset, located at `C:\Users\ri\OneDrive\ai project\data cleaning\titanic\data\train_scaled.csv`)
- **Outputs**:
  - `titanic_logreg_predictions.csv` (test predictions, located at `C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\data\titanic_logreg_predictions.csv`)
  - `confusion_matrix.png` (confusion matrix visualization, saved in the script’s working directory)
- **Documentation**: `titanic_logreg_documentation.md` (detailed project documentation)

## Requirements
- **Python Version**: 3.8 or higher
- **Dependencies**:
  ```bash
  pip install pandas numpy scikit-learn seaborn matplotlib
  ```
- **Dataset**: Place `train_scaled.csv` in `C:\Users\ri\OneDrive\ai project\data cleaning\titanic\data\`

## Installation and Setup
1. Clone or download the project repository.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib
   ```
4. Ensure `train_scaled.csv` is in `C:\Users\ri\OneDrive\ai project\data cleaning\titanic\data\`.

## Usage
1. Navigate to the project directory:
   ```bash
   cd C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\titanic_model_logreg
   ```
2. Run the script:
   ```bash
   python titanic_model_logreg.py
   ```
3. Check outputs:
   - Console: Displays data inspection, accuracy, classification report, and confusion matrix.
   - Files: `titanic_logreg_predictions.csv` (in `C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\data\`) and `confusion_matrix.png` (in the script’s working directory).
4. Review documentation in `titanic_logreg_documentation.md` for detailed methodology.

## Expected Output
Running the script produces:
- Console output with dataset details, accuracy (~0.8212), classification report, and confusion matrix.
- A confusion matrix plot (`confusion_matrix.png`) showing true/false positives and negatives.
- A CSV file (`titanic_logreg_predictions.csv`) with test set predictions.

## Next Steps
- Compare with other models (e.g., Random Forest, XGBoost).
- Perform hyperparameter tuning (e.g., adjust `C` in logistic regression).
- Address class imbalance using techniques like SMOTE or class weights.
- Analyze feature importance to identify key predictors of survival.

## Notes
- The script uses `random_state=42` for reproducibility.
- Accuracy (~0.82) and confusion matrix values may vary slightly with different splits or datasets.
- If the dataset path is incorrect, update `data_path` in `titanic_model_logreg.py`.
- The confusion matrix plot is saved as `confusion_matrix.png` to suit non-interactive execution.

## License
This project is for educational purposes and can be freely used or modified.