import pandas as pd
import numpy as np
import os
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import Scaler
from pandas.tseries.offsets import DateOffset

def create_covariates(start_date=pd.to_datetime('2020-07-03 00:00:00'),end_date=pd.to_datetime('2021-02-19 01:00:00')):
    cov = pd.DataFrame()
    if not (os.path.isfile('./covariates.pkl')):
        for file in os.listdir('../0. Input Data/2020/'):
            df = pd.read_csv("../0. Input Data/2020/"+file, sep=";")
            df['t_1h'] = pd.to_datetime(df['t_1h'])
            cov = pd.concat([df,cov])
        for file in os.listdir('../0. Input Data/2021/'):
            df = pd.read_csv("../0. Input Data/2021/"+file, sep=";")
            df['t_1h'] = pd.to_datetime(df['t_1h'])
            cov = pd.concat([df,cov])

        cov = cov[['t_1h','q']].dropna()
        cov = cov.groupby(['t_1h']).agg({'q':'sum'})
        cov.resample('15T').ffill()
        cov.to_pickle('covariates.pkl')

    else:
        cov = pd.read_pickle('./covariates.pkl')

    past_timeserie = TimeSeries.from_dataframe(cov,fill_missing_dates=True, freq='15T')
    past_timeserie = fill_missing_values(past_timeserie)
    past_timeserie= Scaler().fit_transform(past_timeserie)

    return past_timeserie.drop_before(start_date-DateOffset(minutes=15))

