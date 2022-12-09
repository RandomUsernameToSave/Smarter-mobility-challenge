import pandas as pd
from utils.format_submission import format_submission2
import numpy as np

def import_data():
    print("Step 1/4: Importing and Preparing Data")

    train = pd.read_csv("../0. Input Data/train.csv", sep=",")
    train['date'] = pd.to_datetime(train['date'])
    train['Postcode'] = train['Postcode'].astype(str)

    n = len(train)
    n_train = 0.8*n
    train_station = train

    valid_station = train.loc[n_train:n,]


    test_station = pd.read_csv("../0. Input Data/test.csv", sep=",")
    test_station['date'] = pd.to_datetime(test_station['date'])
    test_station['Postcode'] = test_station['Postcode'].astype(str)
    
    train_area = train_station.groupby(['date', 'area']).agg({'Available': 'sum',
                                                            'Charging': 'sum',
                                                            'Passive': 'sum',
                                                            'Other': 'sum',
                                                            'tod': 'max',
                                                            'dow': 'max',
                                                            'Latitude': 'mean',
                                                            'Longitude': 'mean',
                                                            'trend': 'max'}).reset_index()
    valid_area = valid_station.groupby(['date', 'area']).agg({'Available': 'sum',
                                                            'Charging': 'sum',
                                                            'Passive': 'sum',
                                                            'Other': 'sum',
                                                            'tod': 'max',
                                                            'dow': 'max',
                                                            'Latitude': 'mean',
                                                            'Longitude': 'mean',
                                                            'trend': 'max'}).reset_index()
    test_area = test_station.groupby(['date', 'area']).agg({
        'tod': 'max',
        'dow': 'max',
        'Latitude': 'mean',
        'Longitude': 'mean',
        'trend': 'max'}).reset_index()


    train_global = train_station.groupby('date').agg({'Available': 'sum',
                                                    'Charging': 'sum',
                                                    'Passive': 'sum',
                                                    'Other': 'sum',
                                                    'tod': 'max',
                                                    'dow': 'max',
                                                    'trend': 'max'}).reset_index()
    valid_global = valid_station.groupby('date').agg({'Available': 'sum',
                                                    'Charging': 'sum',
                                                    'Passive': 'sum',
                                                    'Other': 'sum',
                                                    'tod': 'max',
                                                    'dow': 'max',
                                                    'trend': 'max'}).reset_index()
    test_global = test_station.groupby('date').agg({
        'tod': 'max',
        'dow': 'max',
        'trend': 'max'}).reset_index()

    station_features = ['Station', 'tod', 'dow', 'area'] + \
        ['trend', 'Latitude', 'Longitude']  # temporal and spatial inputs
    area_features = ['area', 'tod', 'dow'] + ['trend',
                                            'Latitude', 'Longitude']  # temporal and spatial inputs
    global_features = ['tod', 'dow'] + ['trend']  # temporal input
    targets = ["Available","Charging","Passive","Other"] # targets "Available",

    return (targets,train_station,valid_station,test_station,
        train_area,valid_area,test_area,train_global,valid_global,test_global)