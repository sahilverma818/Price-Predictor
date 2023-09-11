import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump


# Load The Data Frame.
housing = pd.read_csv('DragonRealEstates.csv')

# Finding The Information
# print(housing.info())

# Setting the Blank value of 'CHAS' column to bydefault '0'
housing['CHAS'].fillna(0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for (train_index, test_index) in split.split(housing, housing['CHAS']):
    Strat_train_set = housing.loc[train_index]
    Strat_test_set = housing.loc[test_index]

# To Studing data for better purpose save the splitted data to another file.

training_set = pd.DataFrame(Strat_train_set)
training_set.to_csv('training_data.csv')
testing_data = pd.DataFrame(Strat_test_set)
testing_data.to_csv('testing_data.csv')

housing_train_features = Strat_train_set.drop("MEDV", axis=1)
housing_train_labels = Strat_train_set["MEDV"].copy()
housing_test_features = Strat_test_set.drop("MEDV", axis=1)
housing_test_labels = Strat_test_set["MEDV"].copy()

# Creating Pipeline

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
     #------ add as much as you want ------)
     ('Std_scalar', StandardScaler()),
])

prepared_training_data = my_pipeline.fit_transform(housing_train_features)
prepared_testing_data = my_pipeline.fit_transform(housing_test_features)

# Creating a model

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(prepared_training_data, housing_train_labels)
predicted_value = model.predict(prepared_testing_data)


# Evaluating the Model
rmse = np.sqrt(mean_squared_error(housing_test_labels, predicted_value))
# print(rmse)

# Evaluating at More accurate level
scores = cross_val_score(model, prepared_testing_data, housing_test_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

print("RMSE Scores : ", rmse_scores)
print("Mean :", rmse_scores.mean())
print("Standard Deviation : ", rmse_scores.std())

dump(model, 'DragonRealEstates.joblib')








