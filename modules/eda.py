import os
import math
import heapq
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from statsmodels.tsa.stattools import acf

from modules import preprocessing
from modules import plotting


# create logger
logger = logging.getLogger('logger_eda')
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


def sales_duration(data:DataFrame):
    """EDA: computes duration of activity (i.e.: duration of sales).
    """
    data.day = pd.to_datetime(data.day)
    number_of_days = data.day.max() - data.day.min()
    number_of_years = number_of_days.days / 365
    print(number_of_days.days, 'days')
    print(number_of_years, 'years')

    return 0
    

def value_per_timestep(data:DataFrame, reference:str):
    """EDA: computes number of daily events (i.e.: number of sales per day).
    """
    fig, ax = plt.subplots(figsize=(7,4))
    plt.hist(data.value, color='mediumblue')
    
    ax.set(xlabel = "Value Per Timestep",
        ylabel = "Count",
        title = "Distribution of Value Per Timestep")

    plt.savefig('plots/' + reference + '_value_per_timestep.png')

    return 0
    

def autocorrel(data:DataFrame, lag:int):
    """EDA:
    * computes the autocorrelation function over a period of time corresponding to the specified lag,
    * returns the results values as a list.
    """
    results = list(acf(data, nlags=lag))

    return results


def subsetting(env:dict) -> DataFrame:
    """EDA: for all subsets from `/data/products/` folder
    * identifies statistically legit lags,
    * saves the dataframe of every possible lag by run,
    * returns the dataframe.
    """
    # loads global env variables
    scope = env["eda"]["scope"]
    trigger = env["eda"]["trigger"]
    # order = env["eda"]["order"]
    # declares path and lists to store data
    path = 'data/products/'
    Paths, Lags = [[] for x in range(2)]
    # explores osdir to identify monthly data from others
    for directory in os.listdir(path):
        # if any
        if directory == 'monthly':
            # adapts path
            path = 'data/products/monthly/'
            # and appends data if file is .csv
            for filename in os.listdir(path):
                if filename.endswith('.csv'):
                    # with lag set to 12 as for monthly seasonality
                    Lags.append(12)
                    Paths.append(path + filename)
        # for others
        else:
            # adapts path
            path = 'data/products/daily/'
            # gets filename
            for filename in os.listdir(path):
                # if file is .csv
                if filename.endswith('.csv'):
                    # rebuilds filepath
                    filepath = path + filename
                    # gets data from it
                    data = pd.read_csv(filepath).groupby(['day'])['ordered_item_quantity'].sum().reset_index().ordered_item_quantity
                    n = len(data)
                    # proceeds to autocorrelation computations on data
                    data_acf_values = autocorrel(data, int(math.floor(n/2)-1))
                    # appends lists with top 'scope'-best autocorrelation values & indices
                    lags_acf_values = heapq.nlargest(scope, data_acf_values)
                    lags_acf_indices = []
                    for i in range(1, len(lags_acf_values)):
                        lags_acf_indices.append(data_acf_values.index(lags_acf_values[i]))
                    # and with arbitrary defined lags [n/2-1 & n/3-1]
                    lags_acf_indices.append(int(math.floor(n/2)-1))
                    lags_acf_indices.append(int(math.floor(n/3)-1))
                    lags_acf_indices = [x for x in lags_acf_indices if x >= trigger]
                    for index in lags_acf_indices:
                        Lags.append(index)
                    Paths += len(lags_acf_indices) * [path + filename]
    # compiles data in dict then dataframe
    data_subsets = {
            'path'  : Paths,
            'lag'   : Lags
        }
    subsets = pd.DataFrame(data=data_subsets)
    subsets.to_csv('data/subsets.csv')

    return subsets

    
def main(data:DataFrame, reference:str):
    """EDA : computes value per timestep plots exported as a png file.
    """
    logger.info("EDA...")
    # print info of input data
    # print("\nsales_data info: ", data.info())
    # print("\nsales_data head: ", data.head())
    # computes aggregated value per timestep
    value_per_timestep(data, reference)
    # plots value per timestep
    plotting.time_plot(
        data, 
        reference, 
        'timestep', 
        'value', 
        'Periodic Value Before Diff Transformation'
        )
    logger.info("... done.")
