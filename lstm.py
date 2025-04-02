import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras as tf
from keras.models import Sequential
import datetime
from data_model_operations import prepare_pickle
from graphs import plot_epochs, plot_prediction
from accuracy import get_accuracy

def resave_model(ticker, opt, epochs):
    
    prepare_pickle(ticker)
    
    pickle_file_name = "./Data/pickles/"+ ticker + ".pickle"
    df = pickle.load(open(pickle_file_name,"rb"))

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

    scaler = MinMaxScaler()
    base = scaler.fit_transform(base)
    target = scaler.fit_transform(target)

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
    train = model.fit(X_train, y_train, epochs=epochs, batch_size=32)

    loss = train.history['loss']
    epoch_count = range(1, len(loss) + 1)
    plot_epochs(epoch_count, loss)

    pred = model.predict(X_test)
    plot_prediction(y_test, pred, ticker +' Price Prediction for 7 base length')
    
    pred_tr = scaler.inverse_transform(pred)
    y_test_tr = scaler.inverse_transform(y_test)
    plot_prediction(y_test_tr, pred_tr, ticker +' Price Prediction for 7 base length')

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
    
    str = ticker + ' Price Prediction for 7 base length'
    plot_prediction(df['Close'], df['Prediction'], str)
    
    model_file = "./Data/Model/"+ ticker + "_LSTM.h5"
    scaler_file = "./Data/Model/"+ ticker + "_LSTM_Scaler.pickle"
    predictions_dataframe = "./Data/Model/"+ ticker + "_LSTM_pred.pickle"
    
    pickle.dump(model, open(model_file,"wb"))
    pickle.dump(scaler, open(scaler_file,"wb"))
    pickle.dump(X_test, open(predictions_dataframe,"wb"))
    
    get_accuracy(y_test, pred)
    
def get_prediction(ticker):
    pickle_file_name = "./Data/pickles/"+ ticker + ".pickle"
    model_file = "./Data/Model/"+ ticker + "_LSTM.h5"
    scaler_file = "./Data/Model/"+ ticker + "_LSTM_Scaler.pickle"
    prediction_df = "./Data/Model/"+ ticker + "_LSTM_pred.pickle"
   
    df = pickle.load(open(pickle_file_name,"rb"))
    model = pickle.load(open(model_file,"rb"))
    scaler = pickle.load(open(scaler_file,"rb"))
    x_pred = pickle.load(open(prediction_df,"rb"))

    forecast = model.predict(x_pred)
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
    plot_prediction(df['Close'], df['Prediction'], ticker +  ' Future Price Prediction')

    
get_prediction("DOGE-INR")