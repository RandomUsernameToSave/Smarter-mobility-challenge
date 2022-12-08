from darts import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.utils.missing_values import fill_missing_values, extract_subseries

def hierarchy_create(train_station):
    """
    Crée l'arbre d'hierarchie entre les zones et le total"""
    areas = ['north','east','south','west']

    hierarchy = dict()

    for area in areas :
        hierarchy[area] = ['Total']

    return hierarchy

def hierarchy_dataframe(train,target,area_data,relevant_global):
    """Arguments : train dataset, target : 'available','charging'...
    Return : L timeseries to fit model on 
    """
    L = pd.DataFrame()
    areas = ['north','east','south','west']
    print('ok ici aussi')
    for area in areas:
        q = area_data[area_data['area']== area][[target,'date']]
        q['date'] = pd.to_datetime(q['date'])
        q = q.set_index('date')
        L[area] = q
    t = relevant_global[[target,'date']]
    
    t['date'] = pd.to_datetime(t['date'])
    t = t.set_index('date')
    L['Total'] = t

    start_date = t.index.min()
    end_date = t.index.max()
    time_serie = TimeSeries.from_dataframe(L,fill_missing_dates=True, freq='15T') #,fill_missing_dates=True, freq='15T'
    time_serie.plot()
    plt.show()
    time_serie3 = fill_missing_values(time_serie).drop_after(0.4)#.drop_before(0.8) # remove to train on whole dataset 
    time_serie2 = fill_missing_values(time_serie).drop_before(0.8)

    return time_serie3,start_date,end_date, time_serie2

