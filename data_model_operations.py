import yfinance as yf
import datetime
import pickle
from graphs import plot_collection

def get_crypto_data(ticker):
    end = datetime.date.today()
    start = end - datetime.timedelta(730)
    df = yf.download(ticker, start = start, end = end)
    db_name = "./Data/"+ticker+"_Data.csv"
    df.to_csv(db_name)
    return df

def prepare_pickle(ticker):
    df = get_crypto_data(ticker)
    str1 = "./Data/pickles/"+ticker + ".pickle"
    df_file = open(str1,"wb")
    print(str1)
    pickle.dump(df,df_file)

def get_stored_model(ticker):
    str1 = "./Data/pickles/"+ticker + ".pickle"
    df = pickle.load(open(ticker + ".pickle","rb"))
    plot_collection(ticker,df)
