import pickle
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from pandas.core.frame import DataFrame
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from modules import predicting
from modules import plotting


# create logger
logger = logging.getLogger('logger_modeling')
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


def undo_scaling(y_pred, x_test, scaler_obj, lstm=False): 
    """Modeling: for linear, lstm & arima models based computations
    * processes to revert MinMax scaling,
    * returns the inverse transformed dataframe.
    """
    # reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
    # reshapes for neural network structure needs
    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    # rebuilds test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index],x_test[index]],axis=1))
    # reshapes pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    # inverse transforms
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
    
    return pred_test_set_inverted


def get_scores(predicted_data:DataFrame, original_data:DataFrame, model_name:str, lag:int, scores:dict):
    """Modeling: computes statistical scores of ran model as
    * Root-mean-square deviation (rmse),
    * Mean absolute error (mae),
    * R-Squared (r2).
    """
    rmse = np.sqrt(mean_squared_error(original_data.value[-lag:], predicted_data.predicted_value[-lag:]))
    mae = mean_absolute_error(original_data.value[-lag:], predicted_data.predicted_value[-lag:])
    r2 = r2_score(original_data.value[-lag:], predicted_data.predicted_value[-lag:])
    scores[model_name] = [rmse, mae, r2]

    logger.info('{}_RMSE: {}'.format(model_name, rmse))
    logger.info('{}_MAE: {}'.format(model_name, mae))
    logger.info('{}_R2 Score {}'.format(model_name, r2))

    return scores


def test_predict(
    predicted_variations        : DataFrame, 
    original_data               : DataFrame, 
    lag                         : int, 
    arima                       : bool=False
    ):
    """Modeling: 
    * computes test predictions on last data range of length 'lag' to be compared with actual values, 
    * and returns dataframe.
    """
    # declares iterator and data ranges
    iterator = lag + 1
    original_timesteps = list(original_data[-iterator:].timestep)
    original_values = list(original_data[-iterator:].value)
    # iterates over predicted variations to compute predicted values stored in a dict
    result_list = []
    for index in range(len(predicted_variations)):
        result_dict = {}
        # saves corresponding timestep
        result_dict['timestep'] = original_timesteps[index+1]
        # proceeds
        if arima:
            result_dict['predicted_value'] = int(predicted_variations[index] + original_values[index])
        else:
            if index == 0:
                result_dict['predicted_value'] = int(round(predicted_variations[index][0]) + original_values[index])
                
            else:
                result_dict['predicted_value'] = int(round(predicted_variations[index][0]) + result_list[index-1]['predicted_value'])
        # store result dict in result list
        result_list.append(result_dict)
    # compiles list of results into a dataframe
    df_result = pd.DataFrame(result_list)
    
    return df_result


def run_linear(
    model,
    model_name  : str, 
    data        : DataFrame, 
    X_train, y_train, X_test, scaler_object,
    reference   : str, 
    lag         : int, 
    mode        : str, 
    scores      : dict
    ):
    """Modeling: 
    * computes linear regression on specified data with specified model and lag,
    * saves model,
    * plots results.
    """
    # fits model 
    model.fit(X_train, y_train)
    # runs predictions
    predictions = model.predict(X_test)
    # and saves it as .sav file
    pickle.dump(
    model, 
    open(
        "models/" + reference + "_" + model_name + "_model_lag-" + str(lag) + ".sav", 
        "wb"
        )
    )
    # unscales predictions with the scaler object
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    # incorporates it in the dataset
    predict_test = test_predict(unscaled, data, lag)
    # measures predictions accuracy
    scores = get_scores(predict_test, data, model_name, lag, scores)
    # plots data with predictions
    plotting.plot_results(predict_test, data, reference, model_name, lag, mode)

    return scores


def run_lstm(
    model,
    model_name  : str, 
    data        : DataFrame, 
    X_train, y_train, X_test, scaler_object,
    reference   : str, 
    lag         : int, 
    mode        : str, 
    scores      : dict
    ):
    """Modeling: 
    * runs Long short-term memory model on specified data with specified lag,
    * saves model,
    * plots results.
    """
    # reshapes input data as per neural network structure requirements
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    # creates input layer structure
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), 
                stateful=True))
    # adds 2 hidden layers
    model.add(Dense(1))
    model.add(Dense(1))
    # fits model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(
        X_train, y_train, 
        epochs=200, 
        batch_size=1, 
        verbose=0, 
        shuffle=False
        )
    # generates test predictions
    predictions = model.predict(X_test,batch_size=1)
    # saves model as HDF5 file 'my_model.h5'
    model.save('models/' + reference + '_' + model_name + '_model_lag-' + str(lag) + '.h5')
    # unscales predictions 
    unscaled = undo_scaling(predictions, X_test, scaler_object, lstm=True)
    # incorporates it in the dataset
    predict_test = test_predict(unscaled, data, lag)
    # measures predictions accuracy
    scores = get_scores(predict_test, data, 'LSTM', lag, scores)
    # plots data with predictions
    plotting.plot_results(predict_test, data, reference, 'LSTM', lag, mode)

    return scores


def run_arima(
    model,
    model_name  : str, 
    data        : DataFrame, 
    original_df : DataFrame, 
    reference   : str, 
    lag         : int, 
    mode        : str, 
    scores      : dict
    ):
    """Modeling: runs sarimax model on specified data with specified lag and returns
    * a tuple of sarimax results,
    * an enriched dataframe with predicted values,
    * and predictions results as a list.
    """
    # fits model 
    sar = model.fit()
    # runs predictions
    start = len(data)- lag - 1
    end = len(data) + lag
    dynamic = 10
    data['predicted_value'] = sar.predict(start=start, end=end, dynamic=dynamic)
    # and saves it as .sav file
    sar.save(
    'models/' + reference + '_' + model_name + '_model_lag-' + str(lag) + '.sav'
    )
    # incorporates it in the dataset
    predict_test = data.predicted_value[start+dynamic:end]
    predict_test = predict_test.reset_index(drop=True)
    predict_test = test_predict(predict_test, data, lag, arima=True)
    # measures predictions accuracy
    scores = get_scores(data, original_df, 'arima', lag, scores)
    # plots data with predictions
    data[['stationary', 'predicted_value']].plot(color=['mediumblue', 'Red'])
    sar.plot_diagnostics(figsize=(10, 8))
    plotting.plot_results(predict_test, data, reference, 'ARIMA', lag, mode)

    return scores


# create results dataframe
def create_results_df(reference:str):
    """Modeling: returns a dataframe of compared models statistical scores.
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
    
    return results_df


def main(
    data            : DataFrame, 
    data_arima      : DataFrame, 
    X_train, y_train, X_test, scaler_object,
    lag             : int, 
    reference       : str, 
    mode            : str
    ):
    """Modeling:
    * runs training feature on following models: [LinearRegression, RandomForest, XGBoost, LSTM, ARIMA],
    * updates test results.
    """
    logger.info("Modeling...")
    # declare scores dict to be filled with every model performance results
    scores = {}
    # linear regression
    scores = run_linear(
        LinearRegression()                      ,   # model instance
        'LinearRegression'                      ,   # model name
        data                                    ,   # dataset
        X_train, y_train, X_test, scaler_object ,   # scaled subsets & object
        reference                               ,   # product reference
        lag                                     ,   # identified by preprocessing
        mode                                    ,   # user input as main.args()
        scores                                      # model performance to be updated
        )
    # random forest
    scores = run_linear(
        RandomForestRegressor(
            n_estimators=100, 
            max_depth=20
            )                                   ,
        'RandomForest'                          , 
        data                                    , 
        X_train, y_train, X_test, scaler_object ,
        reference                               , 
        lag                                     , 
        mode                                    , 
        scores
        )
    # xgboost
    scores = run_linear(
        XGBRegressor(
            n_estimators=100, 
            learning_rate=0.2, 
            objective='reg:squarederror'
            )                                   ,
        'XGBoost'                               , 
        data                                    , 
        X_train, y_train, X_test, scaler_object ,
        reference                               , 
        lag                                     , 
        mode                                    , 
        scores
        )
    # lstm
    scores = run_lstm(
        Sequential()                            ,
        'LSTM'                                  , 
        data                                    , 
        X_train, y_train, X_test, scaler_object ,
        reference                               , 
        lag                                     , 
        mode                                    , 
        scores
        )
    # arima
    scores = run_arima(
        sm.tsa.statespace.SARIMAX(
            data.stationary,                        # on stationary data
            order=(1,1,1),                          # thus d = 1 for integration
        trend='c'                                   # with no trend nor any seasonality to be accounted for
            ),
        'ARIMA'                                 , 
        data_arima                              , 
        data                                    , 
        reference                               , 
        lag                                     , 
        mode                                    , 
        scores
        )
    # dump scores
    pickle.dump(
        scores, 
        open(
            "models/" + reference + "_model_scores.p", 
            "wb"
            )
        )
    logger.info("... done.")