# Orders forecasting
Times Series forecasting for retail orders predictions.

## Diagnose

### Datasets
Data subsets creation by product, model, season.

### Exploratory Data Analysis
* autocorrelation,
* stationarity,
* trend,
* lag identification.

### Preprocessing
* loads data,
* groups by day,
* gets stationary,
* generates lag-based data for
    * supervised regression models,
    * arima model.

### Regressions
Computes
* linear regression,
* random forest,
* xgboost,
* lstm.

### Arima
Computes ARIMA modeling.

### Results
Compare results from data passed to all models.

### Predict
Setup a lag-dependent prediction of ordering by product, model, season.

### Train
* Train specific dataset to generate models using calculated lag
```bash
python main.py --dataset <datset_path> --mode train
``` 
* Train specific dataset to generate models using custom lag
```bash
python main.py --dataset <datset_path> --mode train --custom_lag <number>
``` 

### Predict
* Predict loading specific model and using calculated lag
```bash
python main.py --dataset <datset_path> --mode predict --predict_days <number> --model_to_load <model_path>
``` 
* Predict loading specific model and using calculated lag
```bash
python main.py --dataset <datset_path> --mode predict --predict_days <number> --model_to_load <model_path> --custom_lag <number>
``` 

### API
* Enable API
```bash
python main.py --mode api
``` 
* Execute API using calculated lag
```bash
http://127.0.0.1:8080/apiGenForecastingResults?dataset=data/products/<datset_path>&custom_lag=NA
``` 
* Execute API using custom lag
```bash
http://127.0.0.1:8080/apiGenForecastingResults?dataset=data/products/<datset_path>&custom_lag=<number>
``` 

### RUN on all products defined in data/products
Lag and list of products need to be defined in file `data/subsets.csv` like:  
| ,  | Path                           | Lag |  
| -- |:-------------------------------|:--- |  
| 0  | data/products/pull_parfait.csv | 10  |
| 1  | data/products/pull_parfait.csv | 150 |
| 2  | data/products/pull_parfait.csv | 282 | 
| 3  | data/products/sneakers_solides.csv | 10  |
| 4  | data/products/sneakers_solides.csv | 150 |
| 5  | data/products/sneakers_solides.csv | 282 | 
<br>

To train and predict on all products defined in `data/subsets.csv`, run the following command. 
```bash
python run_all_products.py --predict_days <number>
``` 
<br>

### Deploying on server using docker
```bash
docker-compose build --no-cache
docker-compose up
``` 
