
import os
import json
import math
import logging
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler


# create logger
logger = logging.getLogger('logger_preprocessing')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s' , datefmt='%m/%d/%Y %I:%M:%S %p')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


# declare preprocessing methods
def load_data(target_path:str, env:dict, order:str):  
    """Loads data and returns a dataframe.
    """
    logger.info(f'order: {order}') 
    # raises TypeError if input parameter is not string type
    if not isinstance(target_path, str):
        raise TypeError("Input parameter has to be string type")
    # if input parameter is product reference and not data path
    if not '.csv' in target_path:
        # recreates data path
        reference = target_path
        target_path = 'data/products/' + reference + '.csv'
    # else, if input parameter is data path
    else:
        if order == 'M':
            reference = str(target_path)[22:-4]
        else:
            reference = str(target_path)[20:-4]
    # loads dataset from path
    data = pd.read_csv(target_path)
    # loads global variables to uniformize columns names for later processings
    timesteps_column = env["preprocessing"]["data_columns"]["timestep"]
    values_column = env["preprocessing"]["data_columns"]["value"]
    # saves timestep and desired values to be analysed
    data = pd.DataFrame({
        'timestep'    : list(data[timesteps_column]),
        'value'       : list(data[values_column])
    })

    logger.info(f'reference: {reference}')    

    return data, reference


def aggregate(data:DataFrame, reference:str, custom_lag:str, order:str='u') -> DataFrame:
    """Preprocessing: aggregates data as requested by the order
    * 'u' for unitary, whatever the order,
    * 's' for seconds,
    * 'm' for minutes,
    * 'h' for hourly,
    * 'D' for dayly,
    * 'M' for monthly,
    * 'Y' for yearly

    if available.
    """
    data = data.copy()
    status = False
    # if aggregation is queried to first order level
    if order == 'u' or order == 's':
        data = data.groupby('timestep')['value'].sum().reset_index()
        status = True
    # else if timestamp includes down to seconds
    elif len(str(data.iloc[0,0])) > 10:
        # and if aggregation is queried to higher orders, adjusts by cutting chars in string 
        if order == 'm':
            data.timestep = data.timestep.apply(lambda x: str(x)[:-3])
        elif order == 'h':
            data.timestep = data.timestep.apply(lambda x: str(x)[:-6])
        elif order == 'D':
            data.timestep = data.timestep.apply(lambda x: str(x)[:-9])
        elif order == 'M':
            data.timestep = data.timestep.apply(lambda x: str(x)[:-12])
        elif order == 'Y':
            data.timestep = data.timestep.apply(lambda x: str(x)[:-15])
        else:
            raise ValueError("1- No known value for order parameter, must be \n* 'u' for unitary, \n* 's' for seconds,  \n* 'm' for minutes,  \n* 'h' for hours, \n* 'M' for monthly or \n* 'Y' for yearly")
    # else if timestamp includes down to days only
    else:
        # and aggregation is queried to higher order 
        if order == 'D':
            data = data.groupby('timestep')['value'].sum().reset_index()
            status = True
        elif order == 'M':
            data.timestep = data.timestep.apply(lambda x: str(x)[:-3])
            # and if no specifed lag
            if custom_lag == 'NA':
                # sets lag to 12, for implied monthly seasonality
                custom_lag = '12'
        elif order == 'Y':
            data.timestep = data.timestep.apply(lambda x: str(x)[:-6])
        else:
            raise ValueError("2- No known value for order parameter, must be \n* 'u' for unitary, \n* 'D' for daily, \n* 'M' for monthly or \n* 'Y' for yearly")
    # if not aggregated yet
    if not status:
        # process aggregation according to adjusted order level
        data = data.groupby('timestep')['value'].sum().reset_index()
    # and saves dataframe to .csv file
    data.to_csv('data/outputs/' + reference + '_aggregated.csv')

    return data, custom_lag


def get_diff(data:DataFrame, reference:str) -> DataFrame:
    """Preprocessing: 
    * computes value difference between each timestep,
    * adds a results column,
    * returns a stationary time series dataframe.
    """
    # creates column of stationary values
    data['stationary'] = data.value.diff()
    # drops the first empty column always created when .diff() method is processed apparently
    data = data.dropna() 
    # saves dataframe as .csv file
    data.to_csv('data/outputs/' + reference + '_stationary_df.csv')

    return data


def generate_supervised(data:DataFrame, reference:str, lag:str) -> DataFrame:
    """Preprocessing: generates a csv file where each row represents a month and columns
    include sales, the dependent variable, and prior sales for each lag.
    """
    # makes copy of data input
    data = data.copy()
    # creates column for each lag
    iterator = lag + 1
    for i in range(1, iterator):
        col_name = 'lag_' + str(i)
        data[col_name] = data.stationary.shift(i)
    # drops null values & reset index
    data = data.dropna().reset_index(drop=True)
    # saves dataframe to csv
    data.to_csv(
        'data/outputs/' + reference + '_model_df.csv', 
        index=False
        )

    return data


def generate_arima_data(data:DataFrame, reference:str) -> DataFrame:
    """Preprocessing: generates a csv file with a datetime index and a dependent sales column
    for ARIMA modeling.
    """
    # makes copy of data input
    data = data.copy()
    # drop null values & reset index
    data = data.dropna().reset_index(drop=True)
    # saves dataframe to .csv
    data.to_csv('data/outputs/' + reference + '_arima_df.csv')

    return data


def tts(data:DataFrame, lag:int):
    """Preprocessing:
    * splits data into train and tests sets,
    * returns tuple of both sets.
    """
    # drop original columns from supervised data input
    data = data.drop(['value', 'timestep'], axis=1)
    # split supervised data input into train & test subsets
    train, test = data[0:-lag].values, data[-lag:].values

    return train, test


def missing_data(data:DataFrame, order:str) -> DataFrame:
    """Preprocessing: manually adds missing timesteps (if any) in dataset with corresponding values set to 0.
    """
    # converts timesteps data to datetime format for local computations
    data.timestep = pd.to_datetime(data.timestep)
    # creates list of continuous dates between first and last registered dates of the input dataset
    index_timesteps = [t.strftime('%Y-%m-%d') for t in pd.date_range(str(data.timestep.iloc[0]), str(data.timestep.iloc[-1]))]

    # extracts all data columns as lists for iteration
    target_timesteps = [t.strftime('%Y-%m-%d') for t in list(data.timestep)]
    target_values = [v for v in list(data.value)]

    # fills data input with missing dates from index_dates with null values
    timesteps, values = ([] for i in range(2))
    for timestep in index_timesteps:
        if timestep not in target_timesteps:
            timesteps.append(timestep)
            values.append(0)
        else:
            index = target_timesteps.index(timestep)
            timesteps.append(target_timesteps[index])
            values.append(target_values[index])
    
    # builds up the dataframe from the dict from the lists
    data = pd.DataFrame({
    'timestep'  : timesteps,
    'value'     : values
    })

    return data


def concatenate(env:dict) -> DataFrame:
    """Preprocessing: creates new subsets if not already existing by dropping any value strictly lower than trigger.
    """
    # set target path
    path = 'data/products/'
    # set trigger from env global variables
    trigger = env["preprocessing"]["concatenation"]["trigger"]
    # for every file in target path
    for filename in os.listdir(path):
        # if file is a .csv and not already concatenated
        if filename.endswith('.csv') and not filename.startswith('concat_'):
            # sets target file path
            filepath = path + filename
            # creates dataframe and groups rows by timestep
            raw = pd.read_csv(filepath).groupby(['timestep'])['value'].sum().reset_index()
            # drops all rows with value under trigger
            concatenated = raw[raw.ordered_item_quantity >= trigger]
            # saves copy of shortened dataset to target path
            concatenated.to_csv(path + 'concat_' + filename)

    return concatenated 


def get_lag(data:DataFrame, lag:str) -> int:
    """Preprocessing: identifies lag for current dataset.
    """
    # raise TypeError if input parameter is not string type
    if not isinstance(lag, str):
        raise TypeError("'lag' input parameter has to be string type")
    if(lag != 'NA'):
        lag = int(lag)
        logger.info("user has set specific lag to {}".format(lag))
    else:
        lag = int(math.floor(len(data.groupby(['timestep'])['value'].sum())/2)-1)
        logger.info("default lag value set to half the dataset length minus one, here: {}".format(lag))
    
    return lag


def scale_data(train_set:DataFrame, test_set:DataFrame):
    """Modeling: 
    * processes MinMax scaling on train and test sets,
    * returns tuple of X_train, y_train, X_test, y_test and the scaler object.
    """
    # applies Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    # reshapes training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    # reshapes test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)
    # gets X & y train & test series
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()
    
    return X_train, y_train, X_test, y_test, scaler


def main(target:str, custom_lag:str, env:dict, order:str):
    """Preprocessing: computes preprocessing operations
    * loads data,
    * generates periodic dataframes, 
    * adds any missing timesteps to dataset,
    * creates stationarity data by differencing values,
    * creates supervised dataset by shifting values to lag,
    * creates arima dataset,
    * returns enriched dataset, arima dataset, train & test subset, lag & product reference.
    """
    logger.info("Preprocessing...")
    # loads raw data and get dataset reference
    data_raw, reference = load_data(target, env, order)
    # aggregates loaded data
    data_aggregated, custom_lag = aggregate(data_raw, reference, custom_lag, order)
    # gets lag
    lag = get_lag(data_aggregated, custom_lag)
    # setup required order's format
    if order == 'M':
        # if order is monthly, does not process missing_data
        data_enriched = data_aggregated
    else:
        # else, generates missing timesteps
        data_enriched = missing_data(data_aggregated, order) 
    # generates static data
    data_stationary = get_diff(data_enriched, reference)
    # generates dataset for regression models
    data_supervised = generate_supervised(data_stationary, reference, lag)
    # generates dataset for arima model
    data_arima = generate_arima_data(data_stationary, reference)
    # generates train & test subsets
    train, test = tts(data_supervised, lag)
    # processes MinMax scaling on train and test subsets
    (
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        scaler_object 
    ) = scale_data(train, test)
    logger.info("... done.")

    return (
        data_enriched, 
        data_arima, 
        X_train, y_train, X_test, scaler_object,
        lag, 
        reference
        )