import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import datetime
from sklearn.ensemble import RandomForestRegressor
from data_model_operations import prepare_pickle
from graphs import plot_prediction
from accuracy import get_accuracy
    
def resave_model(ticker):
    prepare_pickle(ticker)
    
    pickele_filename = "./Data/pickles/"+ticker+".pickle"
    
    df = pickle.load(open(pickele_filename,"rb"))

    x = np.array(df[['Close','Volume']])
    y = df['Close'].shift(-100)
    y = np.array(y).reshape(-1,1)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)

    x_train_test = x[:-100,:]
    y_train_test = y[:-100]
    x_pred = x[-100:,:]
    y_pred = y[-100:]

    x_train, x_test, y_train, y_test = train_test_split(x_train_test, y_train_test, train_size=0.8, shuffle=False)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    plot_prediction(y_test, pred, 'Actual Price V/S Predicted Price')

    pred = pred.reshape(-1,1)
    pred_tr = scaler.inverse_transform(pred)
    y_test_tr = scaler.inverse_transform(y_test)
    print(pred_tr.shape, y_test_tr.shape)
    plot_prediction(y_test_tr, pred_tr, 'Actual Price V/S Predicted Price')
    
    model_file = "./Data/Model/"+ticker+"_RF.h5"
    scaler_file = "./Data/Model/"+ticker+"_RF_Scaler.pickle"
    prediction_df = "./Data/Model/"+ticker+"_RF_pred.pickle"
    
    pickle.dump(model,open(model_file,"wb"))
    pickle.dump(scaler,open(scaler_file,"wb"))
    pickle.dump(x_pred,open(prediction_df,"wb"))
    
    get_accuracy(y_test, pred)
    
    forecast = model.predict(x_pred)
    forecast = forecast.reshape(-1,1)
    forecast = scaler.inverse_transform(forecast)
    forecast = forecast.reshape(-1)

    df['Prediction'] = np.nan
    last_date = pd.to_datetime(df.index[-1])
    last_sec = last_date.timestamp()
    one_day_sec = 86400 
    next_sec = last_sec + one_day_sec 

    for i in forecast:
        next_date = datetime.datetime.fromtimestamp(next_sec)
        next_sec += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i] 

    df = pd.DataFrame(df)
    str12 = ticker +  ' Price Prediction for 7 base length'
    sns.set_theme(style="darkgrid")
    plot_prediction(df["Close"], df["Prediction"], str12)
    plot_prediction(y_test_tr, pred_tr, 'Actual Price V/S Predicted Price')

def get_prediction(ticker):
    pickle_file = "./Data/pickles/"+ ticker + ".pickle"
    model_file = "./Data/Model/"+ ticker + "_RF.h5"
    scaler_file = "./Data/Model/"+ ticker + "_RF_Scaler.pickle"
    prediction_df = "./Data/Model/"+ ticker + "_RF_pred.pickle"
    
    df = pickle.load(open(pickle_file,"rb"))
    model = pickle.load(open(model_file,"rb"))
    scaler = pickle.load(open(scaler_file,"rb"))
    x_pred = pickle.load(open(prediction_df,"rb"))

    forecast = model.predict(x_pred)
    forecast = forecast.reshape(-1,1)
    forecast = scaler.inverse_transform(forecast)
    forecast = forecast.reshape(-1)

    df['Prediction'] = np.nan
    last_date = pd.to_datetime(df.index[-1])
    last_sec = last_date.timestamp()
    one_day_sec = 86400 
    next_sec = last_sec + one_day_sec 

    for i in forecast:
        next_date = datetime.datetime.fromtimestamp(next_sec)
        next_sec += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i] 

    df = pd.DataFrame(df)
    sns.set_theme(style="darkgrid")
    str12 = ticker +  ' Price Prediction for 7 base length'
    plot_prediction(df["Close"], df["Prediction"], str12)
    

get_prediction("DOGE-INR")