# Auction Sale Price Prediction
This project aims to predict the auction sale prices of various types of heavy machinery and construction equipment using historical auction data. The objective is to develop an accurate model that assists in creating a comprehensive price guide for heavy machinery using machine learning techniques.
The dataset includes detailed information about the machinery, such as manufacturing year, usage hours, and various machine configurations. The goal is to build a model that minimizes the Root Mean Squared Error (RMSE) to achieve a better score in the competition.

### Prerequisites
You will need the following libraries to run the project:
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn

Install the necessary libraries using:  pip install -r requirements.txt

## Data
The data consists of historical auction results provided in two CSV files: `train.csv` (for training the model) and `valid.csv` (for validation and generating predictions).
## Model
The model used is a `RandomForestRegressor` from scikit-learn, which was chosen for its ability to handle complex interactions between features and robustness against overfitting.
## Evaluation Metrics
The model performance is evaluated using the Root Mean Squared Error (RMSE) on both the training and validation sets.

## Steps to Run
1. **Download Data**: The script downloads the required data files from Google Drive.
2. **Data Preprocessing**: Missing values are handled, and non-numeric columns are removed.
3. **Model Training**: The RandomForestRegressor model is trained on the training data.
4. **Feature Importance**: Permutation feature importance is calculated to identify the most impactful features.
5. **Model Interpretation**: LIME is used to explain individual predictions.
6. **Hyperparameter Tuning**: GridSearchCV is used to find the best hyperparameters for the model.
7. **Regularization**: Regularization techniques are applied to reduce overfitting.
8. **Prediction**: The model generates predictions for the validation set, which are saved to a CSV file.

## Files
- `train.csv`: Training data file.
- `valid.csv`: Validation data file.
- `submission_<timestamp>.csv`: Submission file with predictions.
- `Main.py`: Main script file.

## Submission File
The final submission file contains the following columns:
- `SalesID`: Unique identifier of a particular sale.
- `SalePrice`: Predicted sale price in USD.

## Requirements
The following Python packages are required to run the project:
- plaintext
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lime
- gdown

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

## Features
The following features were selected for the model based on their relevance to auction prices:
- `YearMade`: The year the machine was manufactured.
- `MachineHoursCurrentMeter`: Current usage of the machine in hours.
- `ModelID`: Identifier for a unique machine model.
- `UsageBand_Low`: Low usage band compared to average usage for similar machines.
- `UsageBand_Medium`: Medium usage band compared to average usage for similar machines.
- `fiModelDesc`: Description of a unique machine model.
- `fiBaseModel`: Base model of the machine.
- `fiSecondaryDesc`: Secondary description of the model.
- `fiModelSeries`: Series of the model.
- `fiModelDescriptor`: Further disaggregation of the model description.
- `ProductSize`: The size class grouping for a product group.
- `ProductClassDesc`: Description of the second level hierarchical grouping of the model.
- `Engine_Horsepower`: Engine horsepower rating.
- `Transmission`: Type of transmission (automatic or manual).
- `Drive_System`: Machine configuration (e.g., 2 or 4 wheel drive).
- `Enclosure`: Whether the machine has an enclosed cab.
- `Hydraulics`: Type of hydraulics system.
- `Turbocharged`: Whether the engine is naturally aspirated or turbocharged.
- `Blade_Type`: Type of blade.
- `Undercarriage_Pad_Width`: Width of the crawler treads.
- `Stick_Length`: Length of the machine's digging implement.
- `Coupler`: Type of implement interface.








