# Heavy Machinery Auction Price Prediction

## Project Overview

This project aims to predict the auction sale prices of various types of heavy machinery and construction equipment using historical auction data. The objective is to develop an accurate model that assists in creating a comprehensive price guide for heavy machinery.

## Data

The data consists of historical auction results provided in two CSV files: `train.csv` (for training the model) and `valid.csv` (for validation and generating predictions).

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
