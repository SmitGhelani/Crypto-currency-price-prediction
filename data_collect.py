import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def get_graph():
    buffer = BytesIO()
    buffer.flush()
    plt.savefig(buffer, format='png')  #,dpi=700
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_crypto_data(tickers):
    end = datetime.date.today()
    start = end - datetime.timedelta(730)
    global df
    df = yf.download(tickers, start = start, end = end)
    db_name = str(tickers)+"_Data.csv"
    df.to_csv(db_name)
    return df

def prepare_pickle(tickers):
    df = get_crypto_data(tickers)
    str1 = str(tickers) + str(".pickle")
    df_file = open(str1,"wb")
    print(str1)
    pickle.dump(df,df_file)
    # mo_lstm(df,i)
    # mo_rf(df,i)

def get_stored_model(t):
    global df
    str1 = str(t) + str(".pickle")
    df = pickle.load(open(str(t) + str(".pickle"),"rb"))
    plot_collection(t,df)

def plot_collection(tickers,df):
    plt.figure(figsize=(12,6),dpi=110)
    str = "Price of " + tickers
    print(str)
    sns.lineplot(x=df.index,y="Close", data=df).set_title(str)
    #plt.show()
    global graph
    graph = get_graph()

prepare_pickle("DOGE-INR")