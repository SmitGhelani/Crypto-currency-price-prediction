import base64
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

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

def plot_collection(tickers,df):
    plt.figure(figsize=(12,6),dpi=110)
    str = "Price of " + tickers
    print(str)
    sns.lineplot(x=df.index,y="Close", data=df).set_title(str)
    plt.show()
    # global graph
    # graph = get_graph()
    
def plot_epochs(epoch_count, loss):
    plt.figure(figsize=(12,8))
    plt.plot(epoch_count, loss)
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("7 length")
    plt.show()
    
def plot_prediction(original_df, pred_df, title):
    plt.figure(figsize=(12,8), dpi=130)
    plt.plot(original_df, color='blue', label='Real')
    plt.plot(pred_df, color='red', label='Prediction')
    plt.title(title)
    plt.legend()
    plt.show()
    
def get_dynamic_graph():
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
    
    
    #graph2 = get_graph()
    """ fig = go.Figure()
    lst = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig.add_trace(go.Scatter(y=y_test,
                        mode='lines',
                        name='lines'))
    fig.add_trace(go.Scatter(x=pred,
                        mode='lines',
                        name='lines2'))
    fig.show() """