import pickle
from format_submission import format_submission2
import pandas as pd
from darts import TimeSeries
import matplotlib.pyplot as plt



def submit():
    final = pd.DataFrame(columns=['date','area',"Available","Charging","Passive","Other"])
    global_pred =pd.DataFrame(columns=['date',"Available","Charging","Passive","Other"])
    station_pred = pd.read_pickle("./station.pkl")
    test = pd.read_pickle("./test.pkl")
    targets = ["Available","Charging","Passive","Other"]
    forecast = []

    print('Loading previously calculated predictions')
    for target in targets:
        unpickled_df = pd.read_pickle("./"+target+".pkl")
        forecast.append(unpickled_df)

    # Creating "date" column
    for target in forecast:
        target['date']=target.index

    areas = ['north','east','south','west']

    # Creating columns such as : date, Available, Charging, Passive, Other
    for date in forecast[0]['date'].items():
        row_global = [date[0]]
        for forecast_target in forecast:
            row_global.append(forecast_target.loc[date[0],'Total'])
        global_pred.loc[len(global_pred)] = row_global

    # Creating columns such as : date,area, Available, Charging, Passive, Other
    for date in forecast[0]['date'].items():

        for area in areas:
            row = [date[0],area]
            for forecast_target in forecast:
                row.append(forecast_target.loc[date[0],area])
            final.loc[len(final)] = row

    final.drop(final.tail(1).index,inplace=True) 
    global_pred.drop(global_pred.tail(1).index,inplace=True) 

    #Using given functions.
    format_submission2(station_pred,final,global_pred,test)

    print('You can upload your sample results')