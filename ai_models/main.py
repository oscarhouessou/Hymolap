# import libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import linregress

# import keras modules
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# import additional modules
from statistics import mean
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# function to transform time series to supervised learning series
from pandas import DataFrame
from pandas import concat

from functions_lstm import log_dir_name, preprocessing,series_to_supervised,create_lstm_model,create_gru_model,lstm_model,gru_model,load_dataset,performance,plot_real_predict,corr_obs_sim,corr_sim


cal_data =  pd.read_csv('cal.csv')

val_data = pd.read_csv('val.csv')

cal_data.columns = ['Date','Prec','ETP','Debit']
val_data.columns = ['Date','Prec','ETP','Debit']

cal_data.drop(['Date'], inplace =True, axis=1)
val_data.drop(['Date'], inplace =True, axis=1)

#preprocessed dataset
scaled_cal = preprocessing(cal_data)

scaled_val = preprocessing(val_data)

# using function above
reframed_cal = series_to_supervised(scaled_cal, 1, 1)
reframed_val = series_to_supervised(scaled_val, 1, 1)

train = reframed_cal.values
test = reframed_val.values

# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_test, y_test = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

learning_rate = 1e-2
num_lstm_units = 100 
num_gru_units = 100
num_epochs=10
num_batch_size= 100
#training 
#val_loss,history, yhat  = lstm_model(learning_rate, num_lstm_units, num_epochs, num_batch_size,x_train,y_train,x_test)


val_loss,history, yhat  = gru_model(learning_rate, num_gru_units, num_epochs, num_batch_size,x_train,y_train,x_test)


inv_yhat,inv_y, R2,nse, rmse, mae = performance(val_data,yhat)
print(R2,nse, rmse, mae )
# plot predicitons

plot_real_predict(inv_y,inv_yhat)

corr_obs_sim(inv_y,inv_yhat)

corr_sim(inv_yhat)
