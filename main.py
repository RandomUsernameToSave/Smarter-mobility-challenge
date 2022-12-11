from utils.load_data import import_data
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NHiTSModel,BlockRNNModel,TCNModel
from darts import TimeSeries
import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler
from utils.hierarchy import hierarchy_create, hierarchy_dataframe
from utils.covariates import create_covariates
import pickle
from darts.dataprocessing.transformers.reconciliation import TopDownReconciliator,BottomUpReconciliator, MinTReconciliator 
from utils.utils_submit import submit
from utils.benchmark2 import benchmark

benchmark()


scaler = Scaler()

forecast = []
train = pd.read_csv("../0. Input Data/train.csv", sep=",")

(targets,train_station,valid_station,test_station,
train_area,valid_area,test_area,train_global,valid_global,test_global) =import_data()

print('Data imported')

models = []
preds = []

for i, target in enumerate(targets):
    print("==== Target ", target, " ====")
    print("Iteration ", i + 1, "/", len(targets))

    relevant = train[[target] + ['date']+['Station']].dropna()
    relevant_area = train_area[[target] + ['date','area']].dropna()
    relevant_global = train_global[[target] + ['date']].dropna()
    
    #L is the last time serie that we fit the model on,
    #L2 is the first time serie that we fit the model on.

    L2,start_date,end_date,L = hierarchy_dataframe(relevant,target,relevant_area,relevant_global)

    
    #create covariates relative to trafic datas
    past_timeserie = create_covariates(start_date = L.start_time())
    past_timeserie2 = create_covariates(start_date= L2.start_time())

    hierarchy = hierarchy_create(train_station)
    L = L.with_hierarchy(hierarchy)

    ntrain = 300 #len(relevant)
    valid = test_station.dropna()
    nvalid =  2 # len(valid)
        
    # Building and training model

    encoders = {
    #"datetime_attribute": {"past": ["month",'hour','dayofweek']},
    "cyclic": {"past" : ['hour','dayofweek','month']},
    "transformer": Scaler()
    }

    
    scheduler = optim.lr_scheduler.StepLR #ReduceLROnPlateau  StepLR

  
    
        #### Construit le modèle selon les paramètres donnés ####

    Nhits = TCNModel(2000,1921 ,n_epochs =5,
                            kernel_size=10,
                            num_layers=5, #joue peu voir négativement
                            num_filters =10 , # joue beaucoup
                            dilation_base =6,#wtf
                            add_encoders = encoders,
                            loss_fn = nn.L1Loss(), # nn.MSELoss() nn.L1Loss() nn.CrossEntropyLoss() SmapeLoss
                            optimizer_cls = torch.optim.Adam,
                            optimizer_kwargs={'lr': 1e-3},
                            lr_scheduler_cls = scheduler, #a modifier
                            #lr_scheduler_kwargs = {'verbose':True, 'patience' :1}
                            lr_scheduler_kwargs = {'step_size':2,'gamma':0.4,'verbose':True}
    
    )
    
    print('Model Loaded !')
        #### Fit le modèle aux deux jeux de données ####
    Nhits.fit(L2
            ,past_covariates=past_timeserie2
            )
    Nhits.fit(L
            ,past_covariates=past_timeserie
            )
    
    
    naive_forecast = (Nhits.predict(1921,L
    ,past_covariates=past_timeserie
    ))

    # naive_forecast = scaler.inverse_transform(naive_forecast) ## uncomment this part and the one in hierarchy.py to scale data
    

    ## Remove comment to add hierarchy reconciliation

    #reconciliator = MinTReconciliator()  
    #reconciliator.fit(L)
    #naive_forecast = reconciliator.transform(naive_forecast)

    L[['east','Total']].plot(label="data")
    naive_forecast[['east','Total']].plot(label="fitted")


    models.append(Nhits)
    forecast.append(naive_forecast.pd_dataframe())
    naive_forecast.pd_dataframe().to_pickle('./'+target+'.pkl')

#### Replace les données dans le bon format ####
submit()