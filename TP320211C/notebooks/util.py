'''
Utilities for experimenting with dataframes
Just put "import util" on our notebooks
'''
import metnum
import math
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import seaborn as sns


def predict(x, y):
    '''
    Uses our regressor and test against the same set
    TODO: get a better name
    '''
    linear_regressor = metnum.LinearRegression()
    linear_regressor.fit(x,y)
    prediction = linear_regressor.predict(x) # returns a matrix in nx1
    return prediction[:,0] # to vector


def get_errors(real_values, predicted_values):
    '''
    Given two vectors, returns the RMSE and RMSLE
    '''
    rmse = math.sqrt(mean_squared_error(real_values, predicted_values))
    rmsle = math.sqrt(mean_squared_log_error(real_values, predicted_values))
    return (rmse, rmsle)


def clean_dataframe(df, variables):
    '''
    Returns a dataframe with only the rows with valid values
    on the specified variables.

    clean_dataframe(df, ['metroscubiertos', 'provincia'])
    '''
    df2 = df.copy()
    for v in variables:
        df2 = df2[df2[v].notna()]
    return df2


