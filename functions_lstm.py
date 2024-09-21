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
from tensorflow.keras.layers import Dense, LSTM,GRU
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


# define a helper-function for log-dir-name
def log_dir_name(learning_rate, num_lstm_units,
                 num_epochs, num_batch_size):

    # the dir-name for the TensorBoard log-dir
    s = "./5_logs/lr_{0:.0e}_lstm_{1}_epochs_{2}_batch_{3}/"

    # insert all the hyper-parameters in the dir-name
    log_dir = s.format(learning_rate,
                       num_lstm_units,
                       num_epochs,
                       num_batch_size)

    return log_dir

def preprocessing(df):
    # select values of dataframe
    values = df.values

    # remove output data y to rescale later
    values_x = np.delete(values, -1, axis=1)
    values_x.shape

    # rescale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_x = scaler.fit_transform(values_x)
    # select values of output data y
    values_y = values[:, -1]

    # rescale manually output data y
    scaled_y = (values_y -values_y.mean())/ values_y.std()
    # reshape scaled date to concatenate with scaled input data
    scaled_y = scaled_y.reshape((scaled_y.shape[0], 1))

    # concatenate input data x and output data y
    scaled = np.concatenate((scaled_x, scaled_y), axis=1)
    return  scaled

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def create_lstm_model(learning_rate, num_lstm_units, num_batch_size, num_epochs,x_train):
    
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_lstm_units:    Number of lstm units.
    num_epochs:        Number of epochs.
    num_batch_size:    Batch size of model.
    """
                       
    model = Sequential()
    
    model.add(LSTM(units=num_lstm_units, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(units=1))

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def create_gru_model(learning_rate, num_gru_units, num_batch_size, num_epochs,x_train):
    
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_lstm_units:    Number of lstm units.
    num_epochs:        Number of epochs.
    num_batch_size:    Batch size of model.
    """
                       
    model = Sequential()
    
    model.add(GRU(units=num_gru_units, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(units=1))

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model



def lstm_model(learning_rate, num_lstm_units, num_epochs, num_batch_size,x_train,y_train,x_test):
    
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_lstm_units:    Number of lstm units.
    num_epochs:        Number of epochs.
    num_batch_size:    Batch size of model.
    """

  
    # Create the neural network with these hyper-parameters.
    model = create_lstm_model(learning_rate=learning_rate,
                         num_lstm_units=num_lstm_units,
                         num_epochs=num_epochs,
                         num_batch_size=num_batch_size,x_train= x_train)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_lstm_units,
                           num_epochs, num_batch_size)
  
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)
   
    # Use Keras to train the model.
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=num_epochs,
                        batch_size=num_batch_size,
                        validation_split= 0.1,
                        
                        verbose=2, 
                        shuffle=False,
                        callbacks=[callback_log])

    # Get the loss on the validation-set
    # after the last training-epoch.
    val_loss = history.history['val_loss'][-1]
    yhat = model.predict(x_test)
    
    # find a set of hyper-parameters with the LOWEST fitness-value
    return val_loss,history, yhat


def gru_model(learning_rate, num_gru_units, num_epochs, num_batch_size,x_train,y_train,x_test):
    
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_lstm_units:    Number of gru units.
    num_epochs:        Number of epochs.
    num_batch_size:    Batch size of model.
    """

  
    # Create the neural network with these hyper-parameters.
    model = create_lstm_model(learning_rate=learning_rate,
                         num_lstm_units=num_gru_units,
                         num_epochs=num_epochs,
                         num_batch_size=num_batch_size,x_train= x_train)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_gru_units,
                           num_epochs, num_batch_size)
  
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)
   
    # Use Keras to train the model.
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=num_epochs,
                        batch_size=num_batch_size,
                        validation_split= 0.1,
                        
                        verbose=2, 
                        shuffle=False,
                        callbacks=[callback_log])

    # Get the loss on the validation-set
    # after the last training-epoch.
    val_loss = history.history['val_loss'][-1]
    yhat = model.predict(x_test)
    
    # find a set of hyper-parameters with the LOWEST fitness-value
    return val_loss,history, yhat

# load dataset
def load_dataset(cal_data,val_data):
    cal_data.columns = ['Date','Prec','ETP','Debit']
    val_data.columns = ['Date','Prec','ETP','Debit']
    cal_data.drop(['Date'], inplace =True, axis=1)
    val_data.drop(['Date'], inplace =True, axis=1)

    return cal_data,val_data


def performance(val_data,yhat):
    values_y = val_data['Debit'].values
    inv_yhat = yhat * values_y.std() + values_y.mean()

    # get initial values of y_test
    inv_y = val_data['Debit'].iloc[0:-1]

    inv_yhat = inv_yhat.reshape(inv_yhat.shape[0])
    correlation_matrix = np.corrcoef(inv_y, inv_yhat)
    correlation_xy = correlation_matrix[0,1]
    R2 = correlation_xy**2


    # Nash-Sutcliffe Efficiency (NSE) 
    nse = 1 - ( sum((inv_y - inv_yhat) ** 2 ) / sum( (inv_y - mean(inv_y)) ** 2) ) 
    
    # Root Meam Square Error (RMSE)
    rmse = sqrt(1/len(inv_y)* sum((inv_y - inv_yhat) ** 2 ))
    
    #Mean absolute error

    mae = np.sum(np.abs(inv_y - inv_yhat), axis=0) / len(inv_yhat)
    
    return inv_yhat ,inv_y, R2,nse, rmse, mae


def plot_real_predict(X,Y):
    plt.figure(figsize=(8,6))
    plt.plot(X, label='observations')
    plt.plot(Y, label='predictions')
    #plt.title('Predicted vs Observed')
    plt.xlabel('Days', fontsize=16)
    plt.ylabel('Discharge ($m^3/s$)', fontsize=16)
    plt.legend()
    plt.show()

def corr_obs(X,Y):
    
    plt.scatter(X,Y, label='(Q_obs_t,Q_obs_t-1)')
    plt.xlabel("Discharge at t ($m^3/s$)")
    plt.ylabel("Discharge at t-1 ($m^3/s$)")

    plt.legend(fontsize=10)
    
    d = linregress(X,Y)
    print(d)
    Z = np.linspace(0, max(x[0]))
    eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
    plt.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
    plt.legend(fontsize=10)
    plt.show()

def corr_obs_sim(X,Y):
   
    plt.scatter(X,Y, label='(Q_obs,Q_sim)')
    plt.xlabel("Observed discharge ($m^3/s$)")
    plt.ylabel("Simulated discharge ($m^3/s$)")
    plt.legend(fontsize=10)
    
    d = linregress(X,Y)
    #print(d)
    Z = np.linspace(0, max(max(X), max(Y)))
    eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
    plt.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
    plt.legend(fontsize=10)
    plt.show()

def corr_sim(x):
    X = np.array(x[1:len(x)-1])
    Y = np.array(x[0:len(x)-2])
    plt.scatter(X,Y, label='(Q_sim_t,Q_sim_t-1)')
    plt.xlabel("Simulated discharge at t ($m^3/s$)")
    plt.ylabel("Simulated discharge at t-1 ($m^3/s$)")
    plt.legend(fontsize=10)
    
    d = linregress(X,Y)
    #print(d)
    Z = np.linspace(0, max(x))
    eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
    plt.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
    plt.legend(fontsize=10)
    plt.show()