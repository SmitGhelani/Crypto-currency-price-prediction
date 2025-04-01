import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras as tf
from keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import plotly.graph_objects as go
#import data_collect
import datetime
from keras.models import load_model
from tensorflow.python.eager.context import graph_mode
from sklearn.ensemble import RandomForestRegressor

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')  #,dpi=700
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def find_pre(t):
    global df
    str1 = t+".pickle"
    
    df = pickle.load(open(str1,"rb"))

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
    # plt.figure(figsize=(12,8))
    # plt.plot(y_test, color='blue', label='Real')
    # plt.plot(pred, color='red', label='Prediction')
    # plt.title('Actual Price V/S Predicted Price')
    # plt.legend()
    # plt.show()

    pred = pred.reshape(-1,1)
    pred_tr = scaler.inverse_transform(pred)
    y_test_tr = scaler.inverse_transform(y_test)
    print(pred_tr.shape, y_test_tr.shape)
    # plt.figure(figsize=(12,8))
    # plt.plot(y_test_tr, color='blue', label='Real')
    # plt.plot(pred_tr, color='red', label='Prediction')
    # plt.title('Actual Price V/S Predicted Price')
    # plt.legend()
    # plt.show()
    str2 = "./Data/Model/"+t+"_RF.h5"
    str3 = "./Data/Model/"+t+"_RF_Scaler.pickle"
    str4 = "./Data/Model/"+t+"_RF_pred.pickle"
    pickle.dump(model,open(str2,"wb"))
    pickle.dump(scaler,open(str3,"wb"))
    pickle.dump(x_pred,open(str4,"wb"))
    errors = abs(pred - y_test)
    print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
    mape = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    
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
    df['Close'].plot(figsize=(12,8), color='blue', label='Real')
    df['Prediction'].plot(figsize=(12,8), color='red', label='Prediction')
    str12 = str(t) +  ' Price Prediction for 7 base length'
    plt.title(str12)
    plt.legend()
    # plt.plot(y_test_tr, color='blue', label='Real')
    # plt.plot(pred_tr, color='red', label='Prediction')
    # plt.title('Actual Price V/S Predicted Price')
    # plt.legend()
    plt.show()
    # global graph
    # graph = get_graph()
    # fig = go.Figure()

    # fig.add_trace(go.Scatter(y=df['Close'],x=df.index,
    #                     mode='lines',
    #                     name='Real'))
    # fig.add_trace(go.Scatter(y=df['Prediction'],x=df.index,
    #                     mode='lines',
    #                     name='Prediction'))
    # fig.show() 

find_pre("DOGE-INR")