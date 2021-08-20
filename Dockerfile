FROM python:3.8

WORKDIR /usr/src/app


RUN apt-get update && apt-get upgrade -y
RUN pip install tensorflow==2.2
RUN pip install pandas
RUN pip install seaborn
RUN pip install statsmodels
RUN pip install sklearn
RUN pip install xgboost
RUN pip install keras
RUN pip install flask
RUN pip install flask_cors

COPY . .

CMD python main.py --mode api