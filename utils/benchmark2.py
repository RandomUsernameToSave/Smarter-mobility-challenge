import pandas as pd
from format_submission import format_submission2
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from utility import *
# ET SI JE COMBINAIS LE MODELE POUR LES STATIONS PUIS RNN SUR AREA ET GLOBAL
#### 1. Importing and Preparing Data ####

def benchmark():

    print("Step 1/4: Importing and Preparing Data")

    train = pd.read_csv("../0. Input Data/train.csv", sep=",")
    train['date'] = pd.to_datetime(train['date'])
    train['Postcode'] = train['Postcode'].astype(str)

    n = len(train)
    n_train = 0.8*n
    train_station = train#.loc[0:n_train,]
    # print(train_station)
    valid_station = train.loc[n_train:n,]
    # print(valid_station)

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

    #### 2. Target Analysis ####

    print("Step 2/4: Target Analysis")

    station_features = ['Station', 'tod', 'dow', 'area'] + \
        ['trend', 'Latitude', 'Longitude']  # temporal and spatial inputs
    area_features = ['area', 'tod', 'dow'] + ['trend',
                                            'Latitude', 'Longitude']  # temporal and spatial inputs
    global_features = ['tod', 'dow'] + ['trend']  # temporal input
    targets = ["Available","Charging","Passive","Other"] # targets

    #### 3. Modelling ####

    print("Step 3/4: Modelling")

    ### a. Station ###
    print("=== Station ===")

    station_models = catboost_training(
                                    train = train_station,
                                    test = valid_station,
                                    features = station_features,
                                    cat_features = [0,1,2,3],
                                    targets = targets,
                                    learning_rate = 0.1,
                                    classif = False)

    s_pred = catboost_prediction(station_models,
                                test_station,
                                features = station_features,
                                targets=targets,
                                level_col='Station')



    #### 4. Output prediction ####

    print("Step 4/4: Output prediction")
    s_pred.to_pickle('./station.pkl')
    test_station.to_pickle('./test.pkl')
