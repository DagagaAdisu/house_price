# Imports
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Column selections
cat_columns = ['MSSubClass', 'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle',
    'OverallQual', 'OverallCond', 'ExterQual', 'Foundation', 'BsmtQual',
    'BsmtFinType1', 'HeatingQC', 'CentralAir', 'KitchenQual', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual']
num_columns = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
    'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'MoSold', 'YrSold']

features = cat_columns + num_columns
data = pd.read_csv("train.csv")
data["MSSubClass"] = data["MSSubClass"].astype(str)
X = data[features]
y = data.SalePrice

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Preprocessing
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline([
    ("imputer",SimpleImputer(strategy="constant", fill_value="NA")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value = -1))
])

preprocesser = ColumnTransformer([
    ("nums", num_pipe, num_columns),
    ("cat", cat_pipe,cat_columns)
])

final_model = Pipeline(
    [
        ("preprocesser", preprocesser),
        ("regressor", RandomForestRegressor())
    ]
)

# Training 
final_model.fit(X, y)


# Making submission file for kaggle 
test_data = pd.read_csv("test.csv")
test_data["MSSubClass"] = test_data["MSSubClass"].astype(str)
X_val = test_data[features]
y_pred = final_model.predict(X_val)
submission = pd.DataFrame({
   "Id": test_data["Id"],
    "SalePrice": y_pred
})

submission.to_csv("submission.csv", index=False)
