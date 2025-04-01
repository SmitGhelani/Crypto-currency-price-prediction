import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras as tf
from keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import plotly.graph_objects as go
#import data_collect
import data_collect
import datetime

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')  #,dpi=700
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def find_acc(t,opt="adam",e=10):
    data_collect.data_collect(t)
    global df
    df = pickle.load(open("df.pickle","rb"))

    data = df.iloc[:, 3]
    base = []
    target = []
    length = 7

    for i in range(len(data)-length):
        x = data[i:i+length]
        y = data[i+length]
        base.append(x)
        target.append(y)

    base = np.array(base)
    target = np.array(target)
    target = target.reshape(-1,1)
    print(base.shape,target.shape)

    scaler = MinMaxScaler()
    base = scaler.fit_transform(base)
    target = scaler.fit_transform(target)
    # base = MinMaxScaler().fit_transform(base)
    # target = MinMaxScaler().fit_transform(target)

    base = base.reshape((len(base), length, 1))
    print(base.shape)

    X_train = base[:550,:,:]
    X_test = base[550:,:,:]
    y_train = target[:550,:]
    y_test = target[550:,:]

    model = Sequential()
    model.add(tf.layers.LSTM(units=32, return_sequences=True,input_shape=(7,1), dropout=0.2))
    model.add(tf.layers.LSTM(units=32, return_sequences=True,dropout=0.2))
    model.add(tf.layers.LSTM(units=32, dropout=0.2))
    model.add(tf.layers.Dense(units=1))
    print(model.summary())

    model.compile(optimizer=opt, loss='mean_squared_error')
    train = model.fit(X_train, y_train, epochs=e, batch_size=32)

    loss = train.history['loss']
    epoch_count = range(1, len(loss) + 1)
    """ plt.figure(figsize=(12,8))
    plt.plot(epoch_count, loss, )
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("7 length")
    #plt.show() """

    pred = model.predict(X_test)

    plt.figure(figsize=(12,8),dpi=130)
    plt.plot(y_test, color='blue', label='Real')
    plt.plot(pred, color='red', label='Prediction')
    str = t + ' Price Prediction for 7 base length'
    plt.title(str)
    plt.legend()
    #plt.show()
    global graph2
    graph2 = get_graph()
    
    """ fig = go.Figure()
    lst = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig.add_trace(go.Scatter(y=y_test,
                        mode='lines',
                        name='lines'))
    fig.add_trace(go.Scatter(x=pred,
                        mode='lines',
                        name='lines2'))
    fig.show() """

    
    pred_tr = scaler.inverse_transform(pred)
    y_test_tr = scaler.inverse_transform(y_test)
    """ print(pred_tr.shape, y_test_tr.shape)
    plt.figure(figsize=(12,8))
    plt.plot(y_test_tr, color='blue', label='Real')
    plt.plot(pred_tr, color='red', label='Prediction')
    plt.title('Bitcoin Price Prediction for 7 base length')
    plt.legend() """
    #plt.show()

    forecast = model.predict(X_test)
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
    # plt.figure(figsize=(12,8),dpi=100)
    # plt.plot(df['Close'], color='blue', label='Real')
    # plt.plot(df['Prediction'], color='red', label='Prediction')
    # plt.title('Bitcoin Price Prediction for 7 base length')
    # plt.legend()
    # #plt.show()
    # # global graph
    # # graph = get_graph()

    global accuracy
    errors = abs(pred - y_test)
    print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
    global err
    err = round(np.mean(errors), 2)
    mape = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    global acc
    acc = round(accuracy, 2)