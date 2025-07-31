# Titanic Logistic Regression Model

## Project Overview
This project trains a logistic regression model to predict passenger survival on the Titanic using the preprocessed dataset `train_scaled.csv`. The goal is to build a baseline machine learning model for binary classification (survived or not survived), evaluate its performance, and visualize the results. The project includes data loading, train-test split, model training, evaluation with accuracy and confusion matrix, visualization using Seaborn, and saving predictions to a CSV file. Additionally, a bar chart visualizes the distribution of predicted outcomes. This serves as a foundational exercise in machine learning, demonstrating key skills like data preprocessing, model training, evaluation, and visualization.

The project is implemented in Python scripts (`titanic_model_logreg.py` and `plot_predictions_distribution.py`) and documented in a Markdown file (`titanic_logreg_documentation.md`) for portfolio purposes.

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
The main script (`titanic_model_logreg.py`) executes the following steps:
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

A secondary script (`plot_predictions_distribution.py`) generates a bar chart of the predicted `Survived` distribution:
1. Loads `titanic_logreg_predictions.csv`.
2. Calculates the count of each predicted outcome (0 or 1).
3. Creates a bar chart using Seaborn, saved as `predictions_distribution.png`.

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

### Visualizations
1. **Confusion Matrix**:
   ```
                   Predicted
                  | Not Survived | Survived |
   Actual Not Survived |     91      |    14    |
   Actual Survived     |     18      |    56    |
   ```
   A heatmap is saved as `confusion_matrix.png` in the project directory.

2. **Prediction Distribution**:
   A bar chart shows the distribution of predicted outcomes (105 Not Survived, 74 Survived), saved as `predictions_distribution.png`. This visualizes the model’s tendency to predict slightly more non-survivors, consistent with the dataset’s class imbalance.

- **Outputs**:
  - **Plots**: `confusion_matrix.png`, `predictions_distribution.png`
  - **Predictions**: `titanic_logreg_predictions.csv`

## Key Learnings
- **Train-Test Split**: Ensures model evaluation on unseen data, preventing overfitting.
- **Logistic Regression**: A robust baseline for binary classification, effective with preprocessed data.
- **Evaluation Metrics**: Accuracy provides an overall performance measure; the confusion matrix and classification report offer detailed insights into class-specific performance.
- **Visualization**: Seaborn’s heatmap and bar chart enhance interpretability of model performance and prediction distribution.
- **Baseline Models**: Logistic regression serves as a benchmark for comparing more complex models.

## Files
- **Scripts**:
  - `titanic_model_logreg.py` (main script, located at `C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\titanic_model_logreg\titanic_model_logreg.py`)
  - `plot_predictions_distribution.py` (graph script, save in the same directory)
- **Dataset**: `train_scaled.csv` (located at `C:\Users\ri\OneDrive\ai project\data cleaning\titanic\data\train_scaled.csv`)
- **Outputs**:
  - `titanic_logreg_predictions.csv` (test predictions, located at `C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\data\titanic_logreg_predictions.csv`)
  - `confusion_matrix.png` (confusion matrix visualization)
  - `predictions_distribution.png` (prediction distribution bar chart)
- **Documentation**: `titanic_logreg_documentation.md` (detailed project documentation)

## Requirements
- **Python Version**: 3.8 or higher
- **Dependencies**:
  ```bash
  pip install pandas numpy scikit-learn seaborn matplotlib
  ```
- **Dataset**: Place `train_scaled.csv` in `C:\Users\ri\OneDrive\ai project\data cleaning\titanic\data\`
- **Predictions**: Ensure `titanic_logreg_predictions.csv` is in `C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\data\`

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
4. Ensure `train_scaled.csv` and `titanic_logreg_predictions.csv` are in the specified directories.

## Usage
1. Navigate to the project directory:
   ```bash
   cd C:\Users\ri\OneDrive\ai project\model\Titanic Logistic Regression\titanic_model_logreg
   ```
2. Run the main script to train the model and generate predictions:
   ```bash
   python titanic_model_logreg.py
   ```
3. Run the graph script to generate the prediction distribution plot:
   ```bash
   python plot_predictions_distribution.py
   ```
4. Check outputs:
   - Console: Displays data inspection, accuracy, classification report, and confusion matrix (main script); or distribution counts (graph script).
   - Files: `titanic_logreg_predictions.csv`, `confusion_matrix.png`, `predictions_distribution.png`.
5. Review documentation in `titanic_logreg_documentation.md` for detailed methodology.

## Expected Output
- **Main Script**:
  - Console output with dataset details, accuracy (~0.8212), classification report, and confusion matrix.
  - Files: `titanic_logreg_predictions.csv`, `confusion_matrix.png`.
- **Graph Script**:
  - Console output with prediction distribution (e.g., 105 Not Survived, 74 Survived).
  - File: `predictions_distribution.png` (bar chart).

## Next Steps
- Compare with other models (e.g., Random Forest, XGBoost).
- Perform hyperparameter tuning (e.g., adjust `C` in logistic regression).
- Address class imbalance using techniques like SMOTE or class weights.
- Analyze feature importance to identify key predictors of survival.

## Notes
- The scripts use `random_state=42` for reproducibility.
- Accuracy (~0.82) and confusion matrix values may vary slightly with different splits or datasets.
- If file paths are incorrect, update `data_path` in `titanic_model_logreg.py` or `predictions_path` in `plot_predictions_distribution.py`.
- The plots are saved as files to suit non-interactive execution. To display interactively, replace `plt.savefig()` with `plt.show()`.

## License
This project is for educational purposes and can be freely used or modified.