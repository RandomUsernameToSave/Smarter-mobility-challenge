# Smarter-mobility-challenge


## How it works
We will train 2 models on the data, and store prediction in Station.pkl, Available.pkl, Charging.pkl, Passive.pkl, Other.pkl, then we will organize our data to be able to submit them.

1. It trains, predicts, and stores predictions in Station.pkl using benchmark2.py.
2. It prepares data to build time series using load_data and hierarchy.
3. It build and fit TCN models to data 
4. It predicts data at area and total level.
5. It regroups data in one dataframe

![My Image](Tableau.png)

## How to reproduce results 
1. Run : git clone https://github.com/RandomUsernameToSave/Smarter-mobility-challenge.git
2. Run : cd Smarter-mobility-challenge
3. Download and install requirements using pip install -r requirements.txt
4. And run : python main.py

