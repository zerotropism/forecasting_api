import os
import math
import pickle
import logging
import datetime
import statsmodels
import pandas as pd
from csv import writer
from  operator import add
from keras.models import load_model
from pandas.core.frame import DataFrame
from pandas.core.groupby.generic import DataFrameGroupBy

from modules import modeling
from modules import plotting


# create logger
logger = logging.getLogger('logger_predicting')
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


def run_predict(
    predicted_variations        : DataFrame, 
    original_data               : DataFrame, 
    lag                         : int,
    order                       : str,
    arima                       : bool=False
    ):
    """Predicting: 
    * computes predicted values from predicted variations in value, 
    * and returns a dataframe.
    """
    # sets format according to required order of data aggregation
    if order == 'M':
        format = '%Y-%m'
        step = datetime.timedelta(days=31)   
    else:
        format = '%Y-%m-%d'
        step = datetime.timedelta(days=1)
    # converts timesteps data to datetime format for local computations
    original_data.timestep = pd.to_datetime(original_data.timestep)
    # declares data ranges
    original_value = original_data.value.iloc[-1]
    original_timestep = original_data.timestep.iloc[-1] + step
    future_timesteps = [
        t.strftime(format) 
        for t in pd.date_range(
            start=original_timestep, 
            periods=lag, 
            freq=order
            )]
    # iterates over predicted variations to compute predicted values stored in a dict
    result_list = []
    for index in range(len(predicted_variations)):
        result_dict = {}
        result_dict['timestep'] = future_timesteps[index]
        # arima data input are one-dimensional array
        if arima:
            predicted_variation = predicted_variations[index]
        # all others are two-dimensional array
        else:
            predicted_variation = predicted_variations[index][0]
        # at first pass
        if index == 0:
            # ... uses original value to compute first predicted one
            result_dict['predicted_value'] = int(round(predicted_variation + original_value))
        # later on...
        else:
            # ... uses last predicted value to compute next one
            last_predicted_value = result_list[index-1]['predicted_value']
            result_dict['predicted_value'] = int(round(predicted_variation + last_predicted_value))
        # stores current result dict in total results list
        result_list.append(result_dict)
    # compiles list of results into a dataframe
    df_result = pd.DataFrame(result_list)
    
    return df_result


def save_run_predicted(
    data            : DataFrame, 
    reference       : str, 
    model_name      : str, 
    lag             : int, 
    mode            : str, 
    env             : dict,
    order           : str
    ):
    """Predicting: for every prediction gets
    * the next predicted peak value,
    * the corresponding timestamp,
    * the end of run timestamp,
    * then saves them in '/data/results.csv', 
    * and returns them as a list.
    """
    # loads global env variables
    trigger = env["predicting"]["trigger"]
    # sets format according to required order of data aggregation
    if order == 'M':
        format = '%Y-%m'        
    else:
        format = '%Y-%m-%d' 
    # gets predicted peak value & corresponding predicted date
    peak_value = data.predicted_value.max()
    peak_index = list(data.predicted_value).index(peak_value)
    peak_date_to_check = data.timestep[peak_index]
    # assumes peack_date is a datetime obj by default
    peak_date = peak_date_to_check
    # checks if peak_date is a str
    peak_date_check_str = isinstance(peak_date_to_check, str)
    # if peak_date is str, converting to datetime obj
    if(peak_date_check_str):
        logger.debug('peak_date is str')
        peak_date = datetime.datetime.strptime(peak_date_to_check, format)
    else:
        logger.debug('peak_date is datetime obj')
    # gets date when predicted values break trigger
    run = data[peak_index:].reset_index()
    try:
        run_last_value = next(x for x in run.predicted_value if x < trigger)
        run_last_index = list(run.predicted_value).index(run_last_value)
        run_last_date_to_check = run.timestep[run_last_index]
    except StopIteration:
        run_last_date_to_check = list(run.timestep)[-1]
    # assumes run_last_date is an datetime obj by default
    run_last_date = run_last_date_to_check
    # checks if run_last_date is a str
    run_last_date_check_str = isinstance(run_last_date_to_check, str)
    # if run_last_date is str, converting to datetime obj
    if(run_last_date_check_str):
        logger.debug('run_last_date is str')
        run_last_date = datetime.datetime.strptime(run_last_date_to_check, format)
    else:
        logger.debug('run_last_date is datetime obj')
    # gets predicted run total scope in days
    logger.debug('run_last_date: {}'.format(run_last_date))
    logger.debug('peak_date: {}'.format(peak_date))
    run_scope = (run_last_date - peak_date).days
    # compacts results as a list
    runner = reference + '_' + model_name + '_' + mode + '_' + 'lag-' +str(lag)
    results = [runner, peak_value, peak_date, run_scope]

    return results


def run_linear(
    data        : DataFrame,
    model_name  : str,
    X_test, scaler_object,
    lag         : int,
    reference   : str,
    mode        : str,
    env         : dict,
    order       : str
):
    """Predicting:
    * runs prediction feature of the selected linear model,
    * returns reformated original data, predicted data and predicted stats.
    """
    # loads the corresponding model
    model_path = "models/" + reference + "_" + model_name + "_model_lag-" + str(lag) + ".sav"
    model = pickle.load(open(model_path, 'rb'))
    # runs predictions
    predictions = model.predict(X_test)
    # unscales predictions with the scaler object
    unscaled = modeling.undo_scaling(predictions, X_test, scaler_object)
    # generates new predictions
    predicted_data = run_predict(unscaled, data, lag, order)
    # saves predicted run stats
    predicted_stats = save_run_predicted(predicted_data, reference, model_name, lag, mode, env, order)
    # plots predicted run data
    plotting.plot_results(predicted_data, data, reference, model_name, lag, mode)

    return data, predicted_data, predicted_stats


def run_lstm(
    data        : DataFrame,
    model_name  : str,
    X_train, X_test, scaler_object,
    lag         : int,
    reference   : str,
    mode        : str,
    env         : dict,
    order       : str
):
    """Predicting:
    * runs prediction of the Long short-term memory model,
    * returns reformated original data, predicted data and predicted stats.
    """
    # reshapes input data as per neural network structure requirements
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    # loads corresponding model
    model_path = 'models/' + reference + '_' + model_name + '_model_lag-' + str(lag) + '.h5'
    model = load_model(model_path)
    # runs predictions
    predictions = model.predict(X_test, batch_size=1)
    # unscales predictions 
    unscaled = modeling.undo_scaling(predictions, X_test, scaler_object, lstm=True)
    # generates new predictions
    predicted_data = run_predict(unscaled, data, lag, order)
    # saves predicted run stats
    predicted_stats = save_run_predicted(predicted_data, reference, model_name, lag, mode, env, order)
    # plots predicted run data
    plotting.plot_results(predicted_data, data, reference, model_name, lag, mode)

    return data, predicted_data, predicted_stats


def run_arima(
    data        : DataFrame,
    model_name  : str,
    lag         : int,
    reference   : str,
    mode        : str,
    env         : dict,
    order       : str
):
    """Predicting:
    * runs prediction of the arima model,
    * returns reformated original data, predicted data and predicted stats.
    """
    # loads the corresponding model
    model_path = "models/" + reference + "_" + model_name + "_model_lag-" + str(lag) + ".sav"
    model = statsmodels.tsa.statespace.sarimax.SARIMAXResults.load(model_path)
    # runs predictions
    start = len(data)- lag - 1
    end = len(data) + lag
    dynamic = 10
    data['predicted_value'] = model.predict(start=start, end=end, dynamic=dynamic)
    # incorporates it in the dataset
    predicted_data = data.predicted_value[start+dynamic:end]
    predicted_data = predicted_data.reset_index(drop=True)
    predicted_data = run_predict(predicted_data, data, lag, order, arima=True)
    # saves predicted run stats
    predicted_stats = save_run_predicted(predicted_data, reference, model_name, lag, mode, env, order)
    # plots predicted run data
    data[['stationary', 'predicted_value']].plot(color=['mediumblue', 'Red'])
    model.plot_diagnostics(figsize=(10, 8))
    plotting.plot_results(predicted_data, data, reference, model_name, lag, mode)

    return data, predicted_data, predicted_stats


def update_results(
    data            : DataFrame, 
    predicted_data  : DataFrame, 
    predicted_stats : list,
    model           : str,
    results         : dict,
    ):
    """Predicting: 
    * updates global prediction results dictionary for specified model,
    * and returns them as a dict.
    """
    # declares specific model dict
    results[model] = {}
    results[model]['dates'] = []
    results[model]['orders'] = []
    # updates with model values
    for i in range(len(predicted_data)):
        results[model]['dates'].append(str(predicted_data['timestep'][i]))
        results[model]['orders'].append(int(predicted_data['predicted_value'][i]))
    # and predicted_stats
    results[model]['predicted_stats'] = {}
    results[model]['predicted_stats']['run'] = str(predicted_stats[0])
    results[model]['predicted_stats']['peak_value'] = str(predicted_stats[1])
    results[model]['predicted_stats']['start_date'] = str(predicted_stats[2])
    results[model]['predicted_stats']['run_scope'] = str(predicted_stats[3])
    # declares original data dict
    results['original_data'] = {}
    results['original_data']['dates'] = []
    results['original_data']['orders'] = []
    # updates with original values
    for i in range(len(data)):
        results['original_data']['dates'].append(str(data['timestep'][i]))
        results['original_data']['orders'].append(int(data['value'][i]))

    return results


def main(
    data            : DataFrame, 
    data_arima      : DataFrame,
    X_train, X_test, scaler_object,
    lag             : int, 
    reference       : str,
    mode            : str,
    env             : dict,
    order           : str,
    model           : str = "all"
    ):
    """Predicting: 
    * runs predict feature on following models: ['all', 'linear', 'LinearRegression', 'RandomForest', 'XGBoost', 'LSTM', 'ARIMA'],
    * updates predictions results, 
    * and return them as a dict.
    """
    logger.info("Predicting...")
    # creates results dict to be updated with predicting outputs
    results = {}
    # if user wants to run all available prediction models on specified dataset
    if model == "all":
        # then loops on a declared list of linear models...
        for m in ['LinearRegression', 'RandomForest', 'XGBoost']:
            predict_subset, predicted_data, predicted_stats = run_linear(data, m, X_test, scaler_object, lag, reference, mode, env, order)
            results = update_results(predict_subset, predicted_data, predicted_stats, m, results)
        # ... the lstm model...
        predict_subset, predicted_data, predicted_stats = run_lstm(data, 'LSTM', X_train, X_test, scaler_object, lag, reference, mode, env, order)
        results = update_results(predict_subset, predicted_data, predicted_stats, 'LSTM', results)
        # ... and finally the arima
        predict_subset, predicted_data, predicted_stats = run_arima(data_arima, 'ARIMA', lag, reference, mode, env, order)
        results = update_results(predict_subset, predicted_data, predicted_stats, 'ARIMA', results)
    # else if user wants to run all available linear models
    elif model == "linear":
        # then loops on a declared list of linear models only
        for m in ['LinearRegression', 'RandomForest', 'XGBoost']:
            predict_subset, predicted_data, predicted_stats = run_linear(data, m, X_test, lag, reference, mode, env, order)
            results = update_results(predict_subset, predicted_data, predicted_stats, m, results)
    # else if any other discretionary model
    elif model == "lr":
        predict_subset, predicted_data, predicted_stats = run_linear(data, 'LinearRegression', X_test, scaler_object, lag, reference, mode, env, order)
        results = update_results(predict_subset, predicted_data, predicted_stats, 'LinearRegression', results)
    elif model == "rf":
        predict_subset, predicted_data, predicted_stats = run_linear(data, 'RandomForest', X_test, scaler_object, lag, reference, mode, env, order)
        results = update_results(predict_subset, predicted_data, predicted_stats, 'RandomForest', results)
    elif model == "xgb":
        predict_subset, predicted_data, predicted_stats = run_linear(data, 'XGBoost', X_test, scaler_object, lag, reference, mode, env, order)
        results = update_results(predict_subset, predicted_data, predicted_stats, 'XGBoost', results)
    elif model == "lstm":
        predict_subset, predicted_data, predicted_stats = run_lstm(data, 'LSTM', X_train, X_test, scaler_object, lag, reference, mode, env, order)
        results = update_results(predict_subset, predicted_data, predicted_stats, 'LSTM', results)
    elif model == "arima":
        predict_subset, predicted_data, predicted_stats = run_arima(data_arima, 'ARIMA', lag, reference, mode, env, order)
        results = update_results(predict_subset, predicted_data, predicted_stats, 'ARIMA', results)
    else:
        # if no model available mentioned, raise KeyError
        raise KeyError("No model selected among: ['all', 'linear', 'lr', 'rf', 'xgb', 'lstm', 'arima'].")
    logger.info("... done.")

    return results