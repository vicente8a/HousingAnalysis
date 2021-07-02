import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split
# from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

""" Get the data """
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


"""
When you call fetch_housing_data, it creates a datasets/housing directory in your workspace,
downloads the housing.tgz file, and extracts the housing.csv file from this directory
"""
fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


"""
The load_housing_data function returns a pandas DataFrame object.
"""
housing = load_housing_data()

""" Create a Test Set"""
# regular way
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# make a copy of the train set so you can play with it without harming the training set
housing = train_set.copy()

""" Looking for Correlations """
# correlation
corr_matrix = housing.corr()
a = corr_matrix['median_house_value'].sort_values(ascending=False)
# print(a)

# housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

""" Attribute Combinations """
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# Now look at correlation again with the new columns
corr_matrix = housing.corr()
# a = corr_matrix['median_house_value'].sort_values(ascending=False)
# print(a)
# print('\n-----------------------------------------------------\n')

""" Prepare the Data for ML Algorithms"""

housing = train_set.drop('median_house_value', axis=1)  # predictors
housing_labels = train_set['median_house_value'].copy()  # label

# you can fill missing values using SimpleImputer
imputer = SimpleImputer(strategy='median')
# since the median can only be computed on numerical values, drop the text attributes
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)

# you fit the data now you transform
x = imputer.transform(housing_num)
housing_tr = pd.DataFrame(x, columns=housing_num.columns, index=housing_num.index)

# encoding
housing_cat = housing[['ocean_proximity']]
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# print(cat_encoder.categories_)
# print(housing_cat_1hot.toarray())

""" """

""" Pipeline """
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('car', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)

""" Select a Train Model """

# LINEAR REGRESSION
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# DECISION TREE
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# RANDOM FOREST
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


scores1 = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores1)

scores2 = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-scores2)

scores3 = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores3)


def display_scores(s):
    print("Scores: ", s)
    print("Mean: ", s.mean())
    print("Standard Deviation: ", s.std())


print("\n----------------------------------------------------------------\n")


print("Decision Tree Cross Validation:")
display_scores(tree_rmse_scores)
print("\n----------------------------------------------------------------\n")

print("Linear Regression Cross Validation:")
display_scores(lin_rmse_scores)
print("\n----------------------------------------------------------------\n")

print("Random Forest Cross Validation:")
display_scores(forest_rmse_scores)
print("\n----------------------------------------------------------------\n")

""" Grid Search """
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_estimator_)


forest_reg = RandomForestRegressor(n_estimators=30, max_features=6, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
scores3 = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores3)

print("\n----------------------------------------------------------------\n")
print("Random Forest Cross Validation (Again):")
display_scores(forest_rmse_scores)
print("\n----------------------------------------------------------------\n")


plt.show()
