import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tarfile
from six.moves import urllib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

# Define constants and paths
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Function to fetch and load the dataset
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()

# Split the dataset into training and test sets
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Create income categories for stratified splitting
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Prepare the data (drop labels for training)
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

# Add extra features to the dataset
housing_tr["rooms_per_household"] = housing_tr["total_rooms"]/housing_tr["households"]
housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"]/housing_tr["total_rooms"]
housing_tr["population_per_household"]=housing_tr["population"]/housing_tr["households"]

# Prepare categorical data
housing_cat = housing[['ocean_proximity']]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

# Start the top-level MLflow run for the experiment
with mlflow.start_run(run_name="housing_price_prediction_pipeline"):

    # 1. Linear Regression Model
    with mlflow.start_run(nested=True, run_name="linear_regression"):
        # Log model parameters
        params = {'max_features': 7, 'n_estimators': 180}
        mlflow.log_param("params", str(params))

        # Train model and log metrics
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        lin_predictions = lin_reg.predict(housing_prepared)
        lin_rmse = np.sqrt(mean_squared_error(housing_labels, lin_predictions))
        mlflow.log_metric("rmse", lin_rmse)

        # Log model
        mlflow.sklearn.log_model(lin_reg, "linear_regression_model")

    # 2. Decision Tree Model
    with mlflow.start_run(nested=True, run_name="decision_tree_regressor"):
        # Log model parameters
        params = {'max_features': 7, 'n_estimators': 180}
        mlflow.log_param("params", str(params))

        # Train model and log metrics
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)
        tree_predictions = tree_reg.predict(housing_prepared)
        tree_rmse = np.sqrt(mean_squared_error(housing_labels, tree_predictions))
        mlflow.log_metric("rmse", tree_rmse)

        # Log model
        mlflow.sklearn.log_model(tree_reg, "decision_tree_regressor_model")

    # 3. Random Forest Model with Randomized Search
    with mlflow.start_run(nested=True, run_name="random_forest_regressor"):
        # Log model parameters
        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
        mlflow.log_param("param_distribs", str(param_distribs))

        # Randomized Search on Random Forest
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                        n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(housing_prepared, housing_labels)
        best_params = rnd_search.best_params_
        mlflow.log_param("best_params", str(best_params))

        # Log best model performance
        cv_results = rnd_search.cv_results_
        best_rmse = np.sqrt(-cv_results["mean_test_score"].max())  # Best RMSE
        mlflow.log_metric("best_rmse", best_rmse)

        # Log best model
        mlflow.sklearn.log_model(rnd_search.best_estimator_, "random_forest_best_model")

    # Ensure models are properly logged and saved in nested runs
    print("Experiment completed.")
