import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
X = pd.read_csv('/train.csv', index_col='Id')
test_data = pd.read_csv('/test.csv', index_col='Id')

#Drop the rows with a null value in the column 'SalePrice'
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

#Assign column 'SalePrice'to y
y = X.SalePrice 

#Drop the column 'SalePrice' in X
X.drop(['SalePrice'], axis=1, inplace=True)

# Select categorical columns with relatively low cardinality (low number of unique values)
categorical_cols = [cname for cname in X.columns if
                    X[cname].nunique() < 10 and 
                    X[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X.columns if 
                X[cname].dtype in ['int64', 'float64']]

# Select the categorical and numerical columns for X and test_data
selected_cols = categorical_cols + numerical_cols
X = X[selected_cols].copy()
test_data = test_data[selected_cols].copy()

# Import necessary libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Create a pipeline for numerical_columns
numerical_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='median')),
    ('num_scaler', RobustScaler())
])

# Create a pipeline for categorical_columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Apply these pipelines to the columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a XGBoost model
model = XGBRegressor(random_state=0, 
                      learning_rate=0.005, n_estimators=1000,
                      max_depth=4,colsample_bytree=0.5, subsample=0.5)

# Apply the preprocessor to the XGBoost model using Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])

from sklearn.model_selection import cross_val_score

# Use cross_validation for the best results
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(pipeline, X, y,
                              cv=5, n_jobs=-1,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())

# Train and test the model to submit to competition
pipeline.fit(X, y)
preds_test = pipeline.predict(test_data)

# Submit to competition
output = pd.DataFrame({'Id': test_data.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)