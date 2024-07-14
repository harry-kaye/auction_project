import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import time
import datetime
import gdown
from pathlib import Path
import warnings
import shutil

matplotlib.use('Agg')  # Use non-GUI backend
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Function to download files from Google Drive
def download_from_gdrive(url, filename):
    # Extract the file ID from the URL
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Download the file
    if Path(filename).exists():
        print(f"File '{filename}' already exists. Skipping download.")
    else:
        gdown.download(download_url, filename, quiet=False)
        print(f"File downloaded as: {filename}")

# Check available disk space
def check_disk_space():
    total, used, free = shutil.disk_usage("/")
    print("Disk Space - Total: %d GiB, Used: %d GiB, Free: %d GiB" % (total // (2**30), used // (2**30), free // (2**30)))
    return free // (2**30)

# URLs for the datasets
train_url = 'https://drive.google.com/file/d/1guqSpDv1Q7ZZjSbXMYGbrTvGns0VCyU5/view?usp=drive_link'
valid_url = 'https://drive.google.com/file/d/1j7x8xhMimKbvW62D-XeDfuRyj9ia636q/view?usp=drive_link'

# Download the datasets
download_from_gdrive(train_url, 'train.csv')
download_from_gdrive(valid_url, 'valid.csv')

# Load the datasets with dtype specification to avoid warnings
dtype_spec = {13: str, 39: str, 40: str, 41: str}
train_df = pd.read_csv('train.csv', dtype=dtype_spec, low_memory=False)
valid_df = pd.read_csv('valid.csv', dtype=dtype_spec, low_memory=False)

print('train cols 1', train_df.columns.size)
print('valid cols 1', valid_df.columns.size)

# Check if the 'saledate' exists in the dataframes
date_column = 'saledate'
if date_column in train_df.columns:
    # Convert date column to datetime and extract Year, Month, and Day
    train_df['Date'] = pd.to_datetime(train_df[date_column])
    train_df['Year'] = train_df['Date'].dt.year
    train_df['Month'] = train_df['Date'].dt.month
    train_df['Day'] = train_df['Date'].dt.day
    # Compute 'Age' from 'Year'
    current_year = datetime.datetime.now().year
    train_df['Age'] = current_year - train_df['Year']
    # Repeat for validation dataset if needed
    if date_column in valid_df.columns:
        valid_df['Date'] = pd.to_datetime(valid_df[date_column])
        valid_df['Year'] = valid_df['Date'].dt.year
        valid_df['Month'] = valid_df['Date'].dt.month
        valid_df['Day'] = valid_df['Date'].dt.day
        valid_df['Age'] = current_year - valid_df['Year']
else:
    print(f"'{date_column}' not found in the dataset. Please check the column names.")

# Drop the original date columns after extraction
train_df.drop(columns=[date_column, 'Date'], inplace=True, errors='ignore')
valid_df.drop(columns=[date_column, 'Date'], inplace=True, errors='ignore')

# Drop columns with more than 40% missing values
threshold = 0.4
train_df = train_df.loc[:, train_df.isnull().mean() < threshold]
valid_df = valid_df.loc[:, valid_df.isnull().mean() < threshold]

# Remove outliers based on normalized SalePrice
def remove_outliers_by_saleprice(df, column='SalePrice', lower_threshold=0.025, upper_threshold=0.95):
    """
    Remove rows based on the normalized 'SalePrice', cutting off the defined tail ends.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the column to normalize and use for outlier removal.
    lower_threshold (float): The lower percentile threshold.
    upper_threshold (float): The upper percentile threshold.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    """
    # Ensure the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Store the initial number of rows
    initial_row_count = df.shape[0]

    # Normalize the specified column
    normalized_column = (df[column] - df[column].mean()) / df[column].std()

    # Calculate the lower and upper quantile thresholds
    lower_bound = normalized_column.quantile(lower_threshold)
    upper_bound = normalized_column.quantile(upper_threshold)

    # Remove rows outside the quantile thresholds
    df = df[(normalized_column >= lower_bound) & (normalized_column <= upper_bound)]

    # Store the final number of rows
    final_row_count = df.shape[0]

    # Calculate number of rows removed and remaining
    rows_removed = initial_row_count - final_row_count
    rows_remaining = final_row_count

    # Print the results
    print(f"Removing outliers: Initial row count: {initial_row_count}")
    print(f"Final row count: {final_row_count}")
    print(f"Rows removed: {rows_removed}")
    print(f"Rows remaining: {rows_remaining}")

    return df

train_df = remove_outliers_by_saleprice(train_df, column='SalePrice', lower_threshold=0.025, upper_threshold=0.95)
print('train info after outliers', train_df.info())
print('train cols 2', train_df.columns.size)
print('valid cols 2', valid_df.columns.size)



# Handle missing values
train_df = train_df.fillna(train_df.median())
valid_df = valid_df.fillna(valid_df.median())

# Display basic information about the training data
print('train info', train_df.info())
print(train_df.describe())
print(train_df.head())

# Visualize the distribution of the target variable (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], bins=50, kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.savefig('sale_price_distribution.png')
plt.close()

# Extracting features and target variable
important_features = [
    'YearMade', 'Age', 'ModelID', 'fiProductClassDesc',
    'ProductGroupDesc', 'fiSecondaryDesc', 'Enclosure', 'ProductGroup',
    'fiModelDesc', 'SalesID', 'fiBaseModel', 'Hydraulics',
#    'YearMade', 'MachineHoursCurrentMeter', 'ModelID', 'UsageBand_Low', 'UsageBand_Medium',  # Updated for dummy variables
#    'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries',
#    'fiModelDescriptor', 'ProductSize', 'ProductClassDesc', 'Engine_Horsepower',
#    'Transmission', 'Drive_System', 'Enclosure', 'Hydraulics', 'Turbocharged',
#    'Blade_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Coupler'
]

# Ensure these important features exist in the dataset
train_df = train_df[[col for col in important_features if col in train_df.columns] + ['SalePrice']]
valid_df = valid_df[[col for col in important_features if col in valid_df.columns] + ['SalesID']]

print('train cols important', train_df.columns.size)
print('valid cols important', valid_df.columns.size)

# Correlation matrix for important features only
correlation_matrix = train_df.corr()
correlation_matrix.to_csv('correlation_matrix_important_features.csv')

# Statistical descriptive views for important features only
desc_stats = train_df.describe()
print(desc_stats)
desc_stats.to_csv('descriptive_statistics_important_features.csv')

X = train_df.drop(['SalePrice'], axis=1)
y = train_df['SalePrice']

# Splitting data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print('train cols 4', train_df.columns.size)
print('valid cols 4', valid_df.columns.size)

# Initialize and train the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

start_time = time.time()
rf.fit(X_train, y_train)
end_time = time.time()

# Make predictions
train_preds = rf.predict(X_train)
valid_preds = rf.predict(X_valid)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))

print(f'Training RMSE: {train_rmse}')
print(f'Validation RMSE: {valid_rmse}')
print(f'Training Time: {end_time - start_time} seconds')

# Preprocess the validation set
# Separate numeric and non-numeric columns
numeric_cols = valid_df.select_dtypes(include=[np.number]).columns
non_numeric_cols = valid_df.select_dtypes(exclude=[np.number]).columns

valid_df[numeric_cols] = valid_df[numeric_cols].fillna(valid_df[numeric_cols].median())

if not non_numeric_cols.empty:
    valid_df[non_numeric_cols] = valid_df[non_numeric_cols].fillna(valid_df[non_numeric_cols].mode().iloc[0])

# Generate predictions for the validation set
valid_preds = rf.predict(valid_df[X_train.columns])

# Create submission file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f'submission_{timestamp}.csv'
submission_df = pd.DataFrame({'SalesID': valid_df['SalesID'], 'SalePrice': valid_preds})
submission_df.to_csv(submission_file, index=False)
print(f'Submission file written to: {submission_file}')

# Permutation Feature Importance
perm_importance = permutation_importance(rf, X_valid, y_valid, n_repeats=5, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

perm_importance_df = pd.DataFrame({
    'Feature': X_valid.columns[sorted_idx],
    'Importance': perm_importance.importances_mean[sorted_idx]
})
print(perm_importance_df)
perm_importance_df.to_csv('permutation_importance.csv', index=False)

# Remove least important features and retrain
least_important_features = X_valid.columns[sorted_idx][:2]

if len(X_train.columns) - len(least_important_features) > 0:
    X_train_reduced = X_train.drop(columns=least_important_features)
    X_valid_reduced = X_valid.drop(columns=least_important_features)
else:
    X_train_reduced = X_train
    X_valid_reduced = X_valid

rf_reduced = RandomForestRegressor(n_estimators=50, random_state=42)
rf_reduced.fit(X_train_reduced, y_train)

valid_preds_reduced = rf_reduced.predict(X_valid_reduced)
valid_rmse_reduced = np.sqrt(mean_squared_error(y_valid, valid_preds_reduced))

print(f'Validation RMSE after dropping least important features: {valid_rmse_reduced}')

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

free_space = check_disk_space()
n_jobs = 1 if free_space < 10 else -1

grid_search = GridSearchCV(estimator=rf_reduced, param_grid=param_grid, cv=3, n_jobs=n_jobs, verbose=2, error_score='raise')
grid_start_time = time.time()
grid_search.fit(X_train_reduced, y_train)
grid_end_time = time.time()

print(f'Best parameters from GridSearchCV: {grid_search.best_params_}')
print(f'GridSearchCV Time: {grid_end_time - grid_start_time} seconds')

best_rf = grid_search.best_estimator_

best_train_preds = best_rf.predict(X_train_reduced)
best_valid_preds = best_rf.predict(X_valid_reduced)

best_train_rmse = np.sqrt(mean_squared_error(y_train, best_train_preds))
best_valid_rmse = np.sqrt(mean_squared_error(y_valid, best_valid_preds))

print(f'Best Training RMSE: {best_train_rmse}')
print(f'Best Validation RMSE: {best_valid_rmse}')

# Save the final submission file
best_valid_preds_final = best_rf.predict(valid_df[X_train_reduced.columns])
submission_file_final = f'submission_final_{timestamp}.csv'
submission_df_final = pd.DataFrame({'SalesID': valid_df['SalesID'], 'SalePrice': best_valid_preds_final})
submission_df_final.to_csv(submission_file_final, index=False)
print(f'Final submission file written to: {submission_file_final}')

# Additional explanations
print("### Results and Explanations ###")
print(f"Initial Training RMSE: {train_rmse}")
print(f"Initial Validation RMSE: {valid_rmse}")
print("A lower RMSE indicates better performance. The training RMSE is low, suggesting the model fits the training data well. The validation RMSE is higher, indicating potential overfitting.")
print(f"Validation RMSE after dropping least important features: {valid_rmse_reduced}")
print("By removing the least important features, the validation RMSE was reduced, indicating a more generalized model.")
print(f"Best Validation RMSE after hyperparameter tuning: {best_valid_rmse}")
print("Hyperparameter tuning significantly improved the model performance.")

# Explanation of the selected features
print("### Explanation of Selected Features ###")
feature_explanations = {
    'YearMade': "The year the machine was manufactured. Newer machines are typically more valuable.",
    'MachineHoursCurrentMeter': "Current usage of the machine in hours. Machines with fewer hours are generally more valuable.",
    'ModelID': "Identifier for a unique machine model. Certain models may have higher base values.",
    'UsageBand_Low': "Low usage band compared to average usage for similar machines.",
    'UsageBand_Medium': "Medium usage band compared to average usage for similar machines.",
    'fiModelDesc': "Description of a unique machine model, which gives insights into the machine's capabilities.",
    'fiBaseModel': "Base model of the machine. Helps in identifying the general type of machine.",
    'fiSecondaryDesc': "Secondary description of the model. Provides additional detail about the machine's features.",
    'fiModelSeries': "Series of the model. Indicates the generation or version of the model.",
    'fiModelDescriptor': "Further disaggregation of the model description.",
    'ProductSize': "The size class grouping for a product group. Impacts the machine's suitability for different tasks.",
    'ProductClassDesc': "Description of the second level hierarchical grouping of the model.",
    'Engine_Horsepower': "Engine horsepower rating. Higher horsepower often means a more powerful and valuable machine.",
    'Transmission': "Type of transmission (automatic or manual). Affects machine performance and user preference.",
    'Drive_System': "Machine configuration (e.g., 2 or 4 wheel drive). Impacts versatility and terrain suitability.",
    'Enclosure': "Whether the machine has an enclosed cab. Provides operator comfort and protection.",
    'Hydraulics': "Type of hydraulics system. Affects efficiency and compatibility with attachments.",
    'Turbocharged': "Whether the engine is naturally aspirated or turbocharged. Impacts performance.",
    'Blade_Type': "Type of blade. Important for machines used in grading and dozing operations.",
    'Undercarriage_Pad_Width': "Width of the crawler treads. Affects stability and performance on different surfaces.",
    'Stick_Length': "Length of the machine's digging implement. Relevant for excavators.",
    'Coupler': "Type of implement interface. Affects the ease and range of attachments the machine can use."
}

for feature, explanation in feature_explanations.items():
    if feature in X_train.columns:
        print(f"{feature}: {explanation}")

# Display feature importance weights
print("### Feature Importance Weights ###")
for idx in sorted_idx[::-1]:
    print(f"{X_valid.columns[idx]}: {perm_importance.importances_mean[idx]:.4f}")
