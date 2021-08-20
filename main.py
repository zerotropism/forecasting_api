import os
import json
import pickle
import logging
import argparse
import pandas as pd
from flask import Flask
from flask import request
from flask_cors import CORS
from pandas.core.frame import DataFrame

from modules import preprocessing
from modules import eda
from modules import modeling
from modules import predicting
from modules import plotting


# create logger
logger = logging.getLogger('logger_main')
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


# loads global variables
env_path = os.path.join('./', 'ENV.json')
env = json.load(open(env_path, 'r'))

# declare api variables
HOME = os.getenv('HOME')
app = Flask(__name__)
CORS(app)
HOST_IP = '127.0.0.1'
HOST_PORT = 8085


# declare api methods
@app.route('/apiPredict',methods=['GET'])
def apiPredict():
    dataset = request.values.get("dataset")
    custom_lag = request.values.get("custom_lag")
    mode = 'predict'
    model = 'all'
    logger.debug('dataset: {}'.format(dataset))
    logger.debug('custom_lag: {}'.format(custom_lag))

    if 'monthly' in dataset:
        order = 'M'
    else:
        order = 'u'

    final_data_dict = forecast(dataset, custom_lag, mode, model, env, order)

    return final_data_dict


@app.route('/apiTrainAndPredict',methods=['GET'])
def apiTrainAndPredict():
    dataset = request.values.get("dataset")
    custom_lag = request.values.get("custom_lag")
    order = request.values.get("order")
    mode = 'full'
    model = 'all'
    logger.debug('dataset: {}'.format(dataset))
    logger.debug('custom_lag: {}'.format(custom_lag))

    final_data_dict = forecast(dataset, custom_lag, mode, model, env, order)

    return final_data_dict


@app.route('/apiLoadSubsets',methods=['GET'])
def apiLoadSubsets():
    subsets = eda.subsetting(env)

    final_data_dict = {}
    final_data_dict['subsets'] = []

    for i in range(len(subsets)):
        data = str(subsets['path'][i])
        lag = str(subsets['lag'][i])
        final_data_dict['subsets'].append(data + '+' + lag)

    return final_data_dict


# declare final methods
def results(lag:int, reference:str):
    """Plots & exports compared results of ran models on a unique product.
    """
    # loads stored models scores dict
    results_dict = pickle.load(open("models/" + reference + "_model_scores.p", "rb"))
    # converts to dataframe
    results_df = pd.DataFrame.from_dict(
        results_dict, 
        orient='index', 
        columns=['RMSE', 'MAE','R2']
        )
    # sorts by RMSE values and reset index
    results_df = results_df.sort_values(by='RMSE', ascending=False).reset_index()
    print(results_df)
    # plots the compared models results
    plotting.plot_compared_results(results_df, reference, lag)

    return 0


def forecast(dataset_path:str, custom_lag:str, mode:str, model:str, env:dict, order:str):
    """Proceeds to compile all procedures for train forecasting.
    """
    # runs data preprocessing
    (
        data, 
        data_arima, 
        X_train, y_train, X_test, scaler_object,
        lag, 
        reference
     ) = preprocessing.main(dataset_path, custom_lag, env, order)
    data.to_csv('data/outputs/' + reference + '_preprocessed.csv')
    # runs data cleaning & eda
    eda.main(data, reference)
    # then sets a switch case on the predict parameter
    if mode == 'full':
        # runs modeling and predicting procedures with preprocessing outputs
        modeling.main(
            data, 
            data_arima,
            X_train, y_train, X_test, scaler_object,
            lag, 
            reference, 
            'train'
            )
        predictions = predicting.main(
            data, 
            data_arima,
            X_train, X_test, scaler_object,
            lag, 
            reference, 
            'predict',
            env,
            order,
            model
        )
        # compares results
        results(lag, reference)
        return predictions
    elif mode == 'train':
        # runs modeling procedure only
        modeling.main(
            data, 
            data_arima,
            X_train, y_train, X_test, scaler_object,
            lag, 
            reference, 
            mode
            )
        # compares results
        results(lag, reference)
        return 0
    elif mode == 'predict':
        # runs predicting procedure only
        predictions = predicting.main(
            data, 
            data_arima,
            X_train, X_test, scaler_object,
            lag, 
            reference, 
            mode,
            env,
            order,
            model
        )
        # compares results
        results(lag, reference)
        return predictions
    else:
        raise KeyError("User did not provide 'mode' key.")


# if executed from the present file
if __name__ == "__main__":
    # checks existence of required folders
    if not os.path.exists('data/outputs'):
        os.makedirs('data/outputs')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('models'):
        os.makedirs('models')
    # declare args commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        help='dataset to forecast', 
                        type=str, 
                        default='NA')
    parser.add_argument('--mode', 
                        help='can be \"full\" or \"train\" or \"predict\" or \"api\" or \"moredata\"', 
                        type=str, 
                        default='full')
    parser.add_argument('--custom_lag', 
                        help='will take into account user defined specified lag', 
                        type=str, 
                        default='NA')
    parser.add_argument('--predict_days', 
                        help='number of days to predict from last date in dataset', 
                        type=str, 
                        default='NA')
    parser.add_argument('--model', 
                        help='model to load for predictions, can be \"all\" or \"linear\" or \"lr\" or \"rf\" or \"xgb\" or \"lstm\" or \"arima\"', 
                        type=str, 
                        default='all')
    args = parser.parse_args()

    

    # if user wants to increase dataset pool
    if args.mode == 'moredata':
        logger.info("Creation of new subsets...")
        preprocessing.concatenate(env)
        logger.info("... done. Your data has been enriched with trigger-based copy of your own. You can now run a train or predict mode.")
    # else if user requests api env
    elif args.mode == 'api':
        logger.info('Running api mode...')
        app.run(host=HOST_IP, port=HOST_PORT)
        logger.info('... done.')
    # else if mode not recognized
    elif args.mode not in ['full', 'train', 'predict', 'api', 'moredata']:
        logger.error('mode \"{}\" not defined, only \"full\" or \"train\" or \"predict\" or \"api\" or \"moredata\"'.format(args.mode))
    # if user requires selected mode to be processed with all available datasets located in data/products/
    elif args.dataset == 'full':
        logger.info("Running train mode on all available datasets...")
        subsets = eda.subsetting(env)
        for i in range(len(subsets)):
            order = ''
            if('monthly' in str(subsets['path'][i])):
                order = 'M'
            else:
                order = 'u'

            forecast(
                str(subsets['path'][i]), 
                str(subsets['lag'][i]),
                args.mode,
                args.model,
                env,
                order
                )
        logger.info("... done.")
    # else runs standalone forecast mode with user parameters
    else:
        order = ''
        if('monthly' in args.dataset):
            order = 'M'
        else:
            order = 'u'

        forecast(args.dataset, args.custom_lag, args.mode, args.model, env, order)
