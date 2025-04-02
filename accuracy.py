import numpy as np
    
def get_accuracy(original_df, prediction_df):
    errors = abs(prediction_df - original_df)
    print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
    mape = 100 * (errors / original_df)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    accuracy = round(accuracy, 2)