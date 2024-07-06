import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')

# Display basic information
print(train_df.info())
print(train_df.describe())

# Display the first few rows of the dataset
print(train_df.head())

# Visualize the distribution of the target variable (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], bins=50, kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Drop non-numeric columns
numeric_train_df = train_df.select_dtypes(include=[np.number])

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_train_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#2
# Handle missing values by dropping columns with a high percentage of missing data
missing_data = train_df.isnull().sum()
missing_data = missing_data[missing_data > 0]
missing_data_percentage = (missing_data / len(train_df)) * 100
columns_to_drop = missing_data_percentage[missing_data_percentage > 50].index
train_df = train_df.drop(columns=columns_to_drop)

# Drop non-numeric columns
train_df = train_df.select_dtypes(include=[np.number])

# Fill remaining missing values with median
train_df = train_df.fillna(train_df.median())

# Extracting features and target variable
X = train_df.drop(['SalePrice'], axis=1)
y = train_df['SalePrice']

# Splitting data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time

# Initialize the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
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

#4
# Preprocess the validation set
valid_df = valid_df.drop(columns=columns_to_drop)
valid_df = valid_df.select_dtypes(include=[np.number])
valid_df = valid_df.fillna(valid_df.median())

# Generate predictions
valid_preds = rf.predict(valid_df)

# Create submission file
submission_df = pd.DataFrame({'SalesID': valid_df['SalesID'], 'SalePrice': valid_preds})
submission_df.to_csv('submission.csv', index=False)

#5
# Feature importance
feature_importances = rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

print(importance_df.head(10))
