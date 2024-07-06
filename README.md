# Auction Sale Price Prediction

This project aims to predict the auction sale prices of machinery using machine learning techniques. The dataset includes detailed information about the machinery, such as manufacturing year, usage hours, and various machine configurations. The goal is to build a model that minimizes the Root Mean Squared Error (RMSE) to achieve a better score in the competition.

## Getting Started

### Prerequisites

You will need the following libraries to run the project:

- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn

Install the necessary libraries using:

```sh
pip install -r requirements.txt

**Explanation of Each Stage**

Stage 1: Data Loading and EDA
Why: Understand the dataset, visualize distributions, and identify missing values.
What: Loaded the CSV files from Google Drive, visualized the distribution of the target variable, and calculated the correlation matrix.

Stage 2: Data Preprocessing
Why: Clean the data by handling missing values and dropping irrelevant columns to prepare for modeling.
What: Dropped columns with more than 50% missing values, kept only numeric columns, and filled remaining missing values with the median.

Stage 3: Model Training
Why: Train a baseline RandomForestRegressor model to predict auction sale prices.
What: Split the data, trained the model, and calculated RMSE on training and validation sets.

Stage 4: Permutation Feature Importance
Why: Identify the most important features impacting model predictions.
What: Calculated and visualized permutation feature importance, then retrained the model after dropping the least important features.

Stage 5: LIME for Model Interpretation
Why: Provide an explainable AI approach to understand how each feature impacts the model's predictions.
What: Used LIME to explain individual predictions and visualize feature importance.

Stage 6: Hyperparameter Tuning
Why: Optimize the model for better performance by finding the best hyperparameters.
What: Used GridSearchCV to find the best hyperparameters and retrained the model with those parameters.

Stage 7: Regularization
Why: Prevent overfitting by adding constraints to the model.
What: Added parameters like min_samples_split and min_samples_leaf and evaluated the model.

Stage 8: Evaluating Different n_estimators
Why: Understand the effect of the number of trees in the forest on the model's performance.
What: Trained the model with different n_estimators values and evaluated RMSE.

Stage 9: Final Submission
Why: Generate predictions for the validation set and save the results.
What: Created a timestamped submission file, a README file, and a requirements file.

This comprehensive approach ensures that the model is well-tuned and its performance is well-understood, with thorough documentation of each step.





