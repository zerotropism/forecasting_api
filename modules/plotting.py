import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from statsmodels.graphics.tsaplots import plot_acf

from modules import eda


def time_plot(data:DataFrame, reference:str, x_col:str, y_col:str, title:str):
    """Plotting: plots the values and their average over their period of time and save the graph.
    """
    # copy and format data.timestep to datetime
    data = data.copy()
    data.timestep = pd.to_datetime(data.timestep)
    # preps plot
    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(x_col, y_col, data=data, ax=ax, color='mediumblue', label='data values')
    second = data.groupby(data.timestep.dt.year)[y_col].mean().reset_index()
    second.timestep = pd.to_datetime(second.timestep, format='%Y')
    sns.lineplot((
        second.timestep + datetime.timedelta(6*365/12)), 
        y_col, 
        data=second, 
        ax=ax, 
        color='red', 
        label='mean'
        )   
    ax.set(xlabel = "Timestep",
        ylabel = "Values",
        title = title)
    sns.despine()
    # saves to /plots/ as .png
    plt.savefig('plots/' + reference + '_time_plot.png')


def plot_results(data_results:DataFrame, original_data:DataFrame, reference:str, model_name:str, lag:int, mode:str):
    """Plotting: plots results of the specified model on specified data and save the graph.
    """
    # converts timesteps data to datetime format for local computations
    original_data.timestep = pd.to_datetime(original_data.timestep)
    data_results.timestep = pd.to_datetime(data_results.timestep)
    # preps plot
    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(original_data.timestep, original_data.value, data=original_data, ax=ax, 
                label='Original', color='mediumblue')
    sns.lineplot(data_results.timestep, data_results.predicted_value, data=data_results, ax=ax, 
                label='Predicted', color='Red')
    ax.set(
        xlabel = "Timesteps",
        ylabel = "Values",
        title = f"{model_name} Forecasting Prediction"
        )
    ax.legend()
    sns.despine()
    # saves results to /data/outputs/ as csv and plot to /plots/ as png
    data_results.to_csv(f'data/outputs/{reference}_{model_name}_{mode}_lag-{lag}.csv')
    plt.savefig(f'plots/{reference}_{model_name}_{mode}_lag-{lag}.png')


def plot_compared_results(data_results:DataFrame, reference:str, lag:int):
    """Plotting: plots compared models statistical scores.
    """
    # preps plot
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(np.arange(len(data_results)), 'RMSE', data=data_results, ax=ax, 
                label='RMSE', color='mediumblue')
    sns.lineplot(np.arange(len(data_results)), 'MAE', data=data_results, ax=ax, 
                label='MAE', color='Cyan')
    plt.xticks(np.arange(len(data_results)),rotation=45)
    ax.set_xticklabels(data_results['index'])
    ax.set(xlabel = "Model",
        ylabel = "Scores",
        title = "Model Error Comparison")
    sns.despine()
    # saves plot to /plots/ as png
    plt.savefig(f'plots/{reference}_compare_models_lag-{lag}.png')


# def plot_autocorrelation(target):
#     """Plotting: plots the autocorrelation function with autocorrelation as y and lags as x.
#     """
#     # list all available subsets
#     subsets = eda.subsetting(env)
#     # lag axis lenght identification of target dataset
#     lag_ax = int(subsets.loc[subsets['Path'] == "data/outputs/" + target + ".csv", 'Lag']) * 2 + 1
#     # loads the target dataset
#     df = pd.read_csv("data/outputs/" + target + "_daily_data.csv")
#     # cleans the timestamp column
#     df['day'] = pd.to_datetime(df['day'])
#     # plots the autocorrelation
#     blue = sns.color_palette("deep", 8)[0]
#     plot_acf(df["ordered_item_quantity"], lags=lag_ax, color=blue)
#     plt.title("Autocorrelation Plot for Value", fontsize=15)
#     plt.ylabel("Correlation", fontsize=15)
#     plt.xlabel("Lag", fontsize=15)
#     plt.show() 