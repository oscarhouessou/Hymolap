from sqlite3 import Date
from pathlib import Path
from shiny import App, render, ui
from shiny.ui import div, head_content, tags


from modhypmahydro import modhypma, criteria, algorithm
from scipy.stats import linregress

import numpy as np
import matplotlib.pyplot as plt

import asyncio
from datetime import date
from sklearn.model_selection import train_test_split


# import mplcyberpunk

# plt.style.use("cyberpunk")

# mplcyberpunk.add_glow_effects()


from shiny.types import FileInfo
import pandas as pd 


inv_yhat_lstm = None
inv_y_lstm = None
#LSTM AND GRU PACKAGES 

import math
import numpy as np

import tensorflow as tf
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

from functions_lstm import log_dir_name, preprocessing,series_to_supervised,create_lstm_model,lstm_model,load_dataset,performance,plot_real_predict,corr_obs_sim,corr_sim

'''
nse_q_sim = metrics['nse']
nse_q_old = 0
nse_sqrt_q_sim = 0
nse_sqrt_q_old = 0
mae_old = 0
mae_sim = metrics['mae']
rmse_old = 0
rmse_sim = metrics['rmse']
#r2_old = 0
#r2_sim = metrics['r2']
'''

# Split général pour tous les modèles
def split_cal_val(x, cal_start , cal_end , val_start, val_end):
  
  # Cette fonction prend en argument les dates de début et de fin du calage et de la validation
  # Ces dates sont pris en tant que chaine de caractère 
  # Voici un exemple cal_start=  '01/01/1965'

  # La fonction retourne les données de calage et de  validation 
  a = x[x['Dates'] == cal_start].index
  a = int(a.values)
  b = x[x['Dates'] == cal_end].index
  b = int(b.values) 

  cal_data = x.iloc[a:b+1] 

  c = x[x['Dates'] == val_start].index
  c = int(c.values)
  d = x[x['Dates'] == val_end].index
  d = int(d.values) 

  val_data = x.iloc[c:d+1]

  return ( cal_data, val_data)


# Modèle ModHyPMA Déterministe
def get_debits(modypmad_x1, modypmad_x2, modypmad_x3, modypmad_x4, data, cal_start , cal_end , val_start, val_end):

    cal_data, val_data = split_cal_val(data, cal_start , cal_end , val_start, val_end)

    cal_file = cal_data.to_csv('calage.csv')
    val_file = val_data.to_csv('validation.csv')

    mod = modhypma.ModHyPMA(
        [modypmad_x1, modypmad_x2, modypmad_x3, modypmad_x4], 
        'modhypmahydro/csv/beterou_62_72_cal.csv', 
        'modhypmahydro/csv/beterou_73_79_val.csv'
    )


    resultat_calage = mod.calibrate(criteria.combine_mae, algorithm.nelder_mead)

    print("Valeur calage")
    print(resultat_calage["q"])

    debits = resultat_calage['q']

    
    return debits

# Modèles LSTM



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

def get_lstm_debits(data, cal_data,val_data,learning_rate,num_lstm_units,num_epochs,num_batch_size):
    
        
        
        cal_data.drop(['Dates'], inplace =True, axis=1)
        val_data.drop(['Dates'], inplace =True, axis=1)

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

      
        #training 
        val_loss,history, yhat  = lstm_model(learning_rate, num_lstm_units, num_epochs, num_batch_size,x_train,y_train,x_test)
        inv_yhat,inv_y, R2,nse, rmse, mae = performance(val_data,yhat)
        return (inv_yhat,inv_y, R2,nse, rmse, mae)

### gru functions
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



def gru_debits(data, cal_data, val_data, learning_rate, num_gru_units, num_epochs, num_batch_size):
    
    
    cal_date = cal_data['Dates']
    val_date = val_data['Dates']
    cal_data.drop(['Dates'], inplace=True, axis=1)
    val_data.drop(['Dates'], inplace=True, axis=1)

    # preprocessing
    scaled_cal = preprocessing(cal_data)
    scaled_val = preprocessing(val_data)

    # supervised dataset
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

    def create_gru_model(learning_rate, num_gru_units, num_batch_size, num_epochs):
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

    def log_dir_name(learning_rate, num_gru_units, num_epochs, num_batch_size):
        # the dir-name for the TensorBoard log-dir
        s = "./5_logs/lr_{0:.0e}_gru_{1}_epochs_{2}_batch_{3}/"
        # insert all the hyper-parameters in the dir-name
        log_dir = s.format(learning_rate, num_gru_units, num_epochs, num_batch_size)
        return log_dir

    def gru_model(learning_rate, num_gru_units, num_epochs, num_batch_size):
        """
        Hyper-parameters:
        learning_rate:     Learning-rate for the optimizer.
        num_gru_units:    Number of gru units.
        num_epochs:        Number of epochs.
        num_batch_size:    Batch size of model.
        """
        # Create the neural network with these hyper-parameters.
        model = create_gru_model(learning_rate, num_gru_units, num_epochs, num_batch_size)
        # Dir-name for the TensorBoard log-files.
        log_dir = log_dir_name(learning_rate, num_gru_units, num_epochs, num_batch_size)
        callback_log = TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            write_grads=False,
            write_images=False
        )
        # Use Keras to train the model.
        history = model.fit(
            x=x_train,
            y=y_train,
            epochs=num_epochs,
            batch_size=num_batch_size,
            validation_split=0.1,
            verbose=2,
            shuffle=False,
            callbacks=[callback_log]
        )
        # Get the loss on the validation-set after the last training-epoch.
        val_loss = history.history['val_loss'][-1]
        yhat = model.predict(x_test)
        return val_loss, history, yhat

    # Call the gru_model function
    val_loss, history, yhat = gru_model(learning_rate, num_gru_units, num_epochs, num_batch_size)
    inv_yhat,inv_y, R2,nse, rmse, mae = performance(val_data,yhat)

    return (inv_yhat,inv_y, R2,nse, rmse, mae)

#### 



#Evaluation de la performance des modèles 


import math
from math import tanh
import numpy as np
def s_curves1(t, x4):
    """
        Unit hydrograph ordinates for UH1 derived from S-curves.
    """

    if t <= 0:
        return 0
    elif t < x4:
        return (t/x4)**2.5
    else: # t >= x4
        return 1


def s_curves2(t, x4):
    """
        Unit hydrograph ordinates for UH2 derived from S-curves.
    """

    if t <= 0:
        return 0
    elif t < x4:
        return 0.5*(t/x4)**2.5
    elif t < 2*x4:
        return 1 - 0.5*(2 - t/x4)**2.5
    else: # t >= x4
        return 1


def gr4j(precip, potential_evap, params, states = None, return_state = False):
    """
        Generated simulated streamflow for given rainfall and potential evaporation.

        :param precip: Catchment average rainfall.
        :type precip: array(float)
        :param potential_evap: Catchment average potential evapotranspiration.
        :type potential_evap: array(float)
        :param params: X parameters for the model.
        :type params: dictionary with keys X1, X2, X3, X4
        :param states: Optional initial state values.
        :type states: Dictionary with optional keys 'production_store', 'routing_store'.
        :param return_state: If true returns a dictionary containing 'production_store' and 'routing_store'. Default: False.
        :type return_state: boolean

        :return: Array of simulated streamflow.
    """
    if states is None:
        states = {}

    X1 = params['X1']
    X2 = params['X2']
    X3 = params['X3']
    X4 = params['X4']

    nUH1 = int(math.ceil(X4))
    nUH2 = int(math.ceil(2.0*X4))

    uh1_ordinates = [0] * nUH1
    uh2_ordinates = [0] * nUH2

    UH1 = states.get('UH1', [0] * nUH1)
    UH2 = states.get('UH2', [0] * nUH2)

    for t in range(1, nUH1 + 1):
        uh1_ordinates[t - 1] = s_curves1(t, X4) - s_curves1(t-1, X4)

    for t in range(1, nUH2 + 1):
        uh2_ordinates[t - 1] = s_curves2(t, X4) - s_curves2(t-1, X4)

    production_store = states.get('production_store', 0) # S
    routing_store = states.get('routing_store', 0) # R

    qsim = []
    for P, E in zip(precip, potential_evap):

        if P > E:
            net_evap = 0
            scaled_net_precip = (P - E)/X1
            if scaled_net_precip > 13:
                scaled_net_precip = 13
            tanh_scaled_net_precip = tanh(scaled_net_precip)
            reservoir_production = (X1 * (1 - (production_store/X1)**2) * tanh_scaled_net_precip) / (1 + production_store/X1 * tanh_scaled_net_precip)

            routing_pattern = P-E-reservoir_production
        else:
            scaled_net_evap = (E - P)/X1
            if scaled_net_evap > 13:
                scaled_net_evap = 13.
            tanh_scaled_net_evap = tanh(scaled_net_evap)

            ps_div_x1 = (2 - production_store/X1) * tanh_scaled_net_evap
            net_evap = production_store * (ps_div_x1) / \
                    (1 + (1 - production_store/X1) * tanh_scaled_net_evap)

            reservoir_production = 0
            routing_pattern = 0

        production_store = production_store - net_evap + reservoir_production

        percolation = production_store / (1 + (production_store/2.25/X1)**4)**0.25

        routing_pattern = routing_pattern + (production_store-percolation)
        production_store = percolation


        for i in range(0, len(UH1) - 1):
            UH1[i] = UH1[i+1] + uh1_ordinates[i]*routing_pattern
        UH1[-1] = uh1_ordinates[-1] * routing_pattern

        for j in range(0, len(UH2) - 1):
            UH2[j] = UH2[j+1] + uh2_ordinates[j]*routing_pattern
        UH2[-1] = uh2_ordinates[-1] * routing_pattern

        groundwater_exchange = X2 * (routing_store / X3)**3.5
        routing_store = max(0, routing_store + UH1[0] * 0.9 + groundwater_exchange)

        R2 = routing_store / (1 + (routing_store / X3)**4)**0.25
        QR = routing_store - R2
        routing_store = R2
        QD = max(0, UH2[0]*0.1+groundwater_exchange)
        Q = QR + QD

        qsim.append(Q)

    if return_state:
        return qsim, {
            'production_store': production_store,
            'routing_store': routing_store,
            'UH1': UH1,
            'UH2': UH2,
        }
    else:
        return qsim

def mean_absolute_error(y_true, y_pred):
   result = np.sum(np.abs(y_pred - y_true), axis=0) / len(y_true)
   return result

def mean_squared_error( y_true, y_pred):
   result = np.sum((y_pred - y_true) ** 2, axis=0) / len(y_true)
   return result

def root_mean_squared_error( y_true, y_pred):
   result = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=0) / len(y_true))
   return result

def  median_absolute_error( y_true, y_pred):
   result = np.median(np.abs(y_true - y_pred), axis=0)
   return result
def nash_sutcliffe_efficiency( y_true, y_pred):
   result = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)   
   return result
def coefficient_of_determination( y_true, y_pred):
    result =  np.sum((y_pred - np.mean(y_pred, axis=0)) ** 2, axis=0) / np.sum((y_true - np.mean(y_pred, axis=0)) ** 2, axis=0)
    return result
   




def gr4j_debit(cal_data,val_data, params):
    # Convert 'Dates' column to the desired format
    

    # Generate simulated flow using GR4J model
    simulated_flow = gr4j(val_data['Prec'], val_data['ETP'], params)

    # Adjust the slicing for dates
    

    # Calculate the metrics
    mae = mean_absolute_error(val_data['Debit'],simulated_flow)
    mse = mean_squared_error(val_data['Debit'],simulated_flow)
    rmse = root_mean_squared_error(val_data['Debit'],simulated_flow)
    nse = nash_sutcliffe_efficiency(val_data['Debit'],simulated_flow)
    r2 = coefficient_of_determination(val_data['Debit'],simulated_flow)
    
    return  simulated_flow,mae, mse, rmse, nse, r2




def panel_box(*args, **kwargs):
    return ui.div(
        ui.div(*args, class_="col-lg-9 p-4 m-0 bg-light-"),
        **kwargs,
    )


app_ui = ui.page_fluid(
    head_content(
        tags.meta(name="viewport", content="width=device-width, initial-scale=1.0"),
        tags.style((Path(__file__).parent / "www/css/style.css").read_text()),
    ),
    ui.div( {"class": "row"},
        ui.div({"class": "shadow p-0 "},
            ui.div( 
                div("ModHyLog : Hydrological Modeling Software", class_="text-5 fw-600 text-center w-100 text-white-"),

                class_="bg-second- hstack px-4 py-2",
            ),
        ),

        ui.navset_tab_card( 
            ui.nav_control(
                ui.a("CIPMA", href="http://www.cipma.net/", target="_blank")
            ),
            ui.nav_control(
                ui.a("LRSIA", href="https://lrsia.ifri-uac.bj", target="_blank")
            ),
            ui.nav_control(
                ui.a("LHA", href="https://c2ea.ine-uac.net/laboratoires", target="_blank")
            ),
            ui.nav_spacer(),

            ui.nav("ModHyPMAD", 

                ui.div(
                    ui.div(
                        ui.div(
                            ui.input_select(
                                "select",
                                ui.span("Choose a dataset:", class_="text-black-"), 
                                choices = {1 : "Beterou basin", 2: "Save basin"}, selected = 2
                            ),
                            ui.input_file("file", 
                                ui.span('Select your dataset', class_="text-6"), 
                                accept=[".csv"], multiple=False,
                            ),
                            ui.hr(),
                            
                                ui.div(
                                   
                                    
                                    ui.div(
                                        ui.h4("Parameters values: ", class_="text-black- text-5"),
                                        ui.input_slider(
                                            "modypmad_x1", 
                                            ui.span("X1 (production store capacity)", class_="text-5 mb-0"),
                                            min = 0, max = 2, animate = True, value = 1.12, step = 0.01, post=" [mm]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_x2", 
                                            ui.span("X2 (groundwater exchange coeff.", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 100, animate = True, value = 35.58, step = 0.01, post=" [mm/d]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_x3", 
                                            ui.span("X3 (routing store capacity)", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 1, animate = True, value = 0.15, step = 0.01, post=" [mm]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_x4", 
                                            ui.span("X4 (unit hydrograph time constant)", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 1, animate = True, value = 0.42, step = 0.01, post=" [d]"
                                        )   
                                    )
                                

                               
                            ),

                            ui.hr(),
  
                            ui.div(
                                ui.h4("Automatic calibration:", class_="text-black- text-5"),
                                ui.input_select(
                                    "select", 
                                    ui.span("Objective function", class_="text-5 mb-0"),
                                    choices= {"NSE(Q)": "NSE(Q)", "MAE": "MAE", "RMSE": "RMSE", "R²": "R²", "NSE[sqrt(Q)]": "NSE[sqrt(Q)]"}
                                ), 
                            ),
                            ui.input_action_button(
                                "run", "Run simulation", class_="btn-primary bg-primary- w-100"
                            ),
                        ),           
                        class_="col-lg-3 p-4 m-0 bg-light-"
                    ),

                    ui.div(
                        ui.div( {"class": "row"},
                            ui.div(
                                ui.row( {"class": "gx-2"},
                                    ui.column(8,
                                        ui.input_slider(
                                            "choose_model",
                                            ui.h6("Select the time window:", class_="text-5 text-black-"),
                                            min=Date(2000, 1, 1),
                                            max=Date(2022, 12, 1),    
                                            step=1,                            
                                            value=[Date(2016, 1, 1), Date(2016, 5, 1)], drag_range=True,
                                            time_format="%Y-%m-%d"
                                        ),
                                        class_="d-none"
                                    ),
                                    ui.column(6,                                        
                                        ui.input_date_range(
                                            "daterange_calage", "Calage:", start=Date(1965, 1, 1), end=Date(1972, 12, 30), language="fr"
                                        ),
                                    ),
                                    ui.column(6,
                                        ui.input_date_range(
                                            "daterange_validation", "Validation:", start=Date(1972, 12, 31), end=Date(1979, 12, 30), language="fr"
                                        ),
                                    ),
                                ),

                                ui.div(
                                    ui.output_plot("plot"),
                                    ui.download_button("download_plot", "Download"),
                                    ui.output_plot("corr_obs"),
                                    ui.output_plot("corr_sim"),
                                    ui.output_plot("corr_obs_sim"),
                                ),

                                class_="col-lg-9 py-2 px-4 bg-white"
                            ),
                            ui.div(
                                ui.div(
                                    ui.row(
                                        ui.column(4, "Criteria"),
                                        ui.column(4, "Qsim"),
                                        ui.column(4, "Qold"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[Q]"),
                                        ui.column(4, "0.28"),
                                        ui.column(4, "0.81"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[sqrt(Q)]"),
                                        ui.column(4, "0.52"),
                                        ui.column(4, "0.63"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "MAE"),
                                        ui.column(4, "0.4"),
                                        ui.column(4, "0.36"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "RMSE"),
                                        ui.column(4, "0.35"),
                                        ui.column(4, "0.45"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "R²"),
                                        ui.column(4, "0.85"),
                                        ui.column(4, "0.64"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                ),  
                                class_="col-lg-3 p-2 py-5 bg-light border text-6 shadom-sm mb-2"
                            )
                        ),
                        class_="col-lg-9 m-0 bg-white"
                    ),

                    class_="row"
                )

            ),
            ui.nav("ModHyPMAS",
                    ui.div(
                        ui.div(                       
                                     
                            ui.div(
                                ui.h4("Parameters values: ", class_="text-black- text-5"),
                                ui.input_slider(
                                    "modypmad_x1", 
                                    ui.span("X1 (production store capacity)", class_="text-5 mb-0"),
                                    min = 0, max = 2, animate = True, value = 1.12, step = 0.01, post=" [mm]"
                                ),
                                ui.input_slider(
                                    "modypmad_x2", 
                                    ui.span("X2 (groundwater exchange coeff.", class_="text-5 my-2 text-black-"),
                                    min = 0, max = 100, animate = True, value = 35.58, step = 0.01, post=" [mm/d]"
                                ),
                                ui.input_slider(
                                    "modypmad_x3", 
                                    ui.span("X3 (routing store capacity)", class_="text-5 my-2 text-black-"),
                                    min = 0, max = 1, animate = True, value = 0.15, step = 0.01, post=" [mm]"
                                ),
                                ui.input_slider(
                                    "modypmad_x4", 
                                    ui.span("X4 (unit hydrograph time constant)", class_="text-5 my-2 text-black-"),
                                    min = 0, max = 1, animate = True, value = 0.42, step = 0.01, post=" [d]"
                                )   
                            )
                          
                    ,           
                        class_="col-lg-3 p-4 m-0 bg-light-"
                    ),

                    ui.div(
                        ui.div( {"class": "row"},
                            ui.div(
                                ui.row( {"class": "gx-2"},
                                    ui.column(8,
                                        ui.input_slider(
                                            "choose_model",
                                            ui.h6("Select the time window:", class_="text-5 text-black-"),
                                            min=Date(2000, 1, 1),
                                            max=Date(2022, 12, 1),    
                                            step=1,                            
                                            value=[Date(2016, 1, 1), Date(2016, 5, 1)], drag_range=True,
                                            time_format="%Y-%m-%d"
                                        ),
                                        class_="d-none"
                                    ),
                                    ui.column(6,                                        
                                        ui.input_date_range(
                                            "daterange_calage", "Calage:", start=Date(1965, 1, 1), end=Date(1972, 12, 30), language="fr"
                                        ),
                                    ),
                                    ui.column(6,
                                        ui.input_date_range(
                                            "daterange_validation", "Validation:", start=Date(1972, 12, 31), end=Date(1979, 12, 30), language="fr"
                                        ),
                                    ),
                                ),

                               ui.div(
                                 ui.output_plot("stoch_plot"),
                                #    ui.download_button("download_plot", "Download"),
                                 #   ui.output_plot("corr_obs"),
                                  #  ui.output_plot("corr_sim"),
                                 #   ui.output_plot("corr_obs_sim"),
                                ),

                                class_="col-lg-9 py-2 px-4 bg-white"
                            ),
                            ui.div(
                                ui.div(
                                    ui.row(
                                        ui.column(4, "Criteria"),
                                        ui.column(4, "Qsim"),
                                        ui.column(4, "Qold"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[Q]"),
                                        ui.column(4, "0.28"),
                                        ui.column(4, "0.81"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[sqrt(Q)]"),
                                        ui.column(4, "0.52"),
                                        ui.column(4, "0.63"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "MAE"),
                                        ui.column(4, "0.4"),
                                        ui.column(4, "0.36"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "RMSE"),
                                        ui.column(4, "0.35"),
                                        ui.column(4, "0.45"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "R²"),
                                        ui.column(4, "0.85"),
                                        ui.column(4, "0.64"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                ),  
                                class_="col-lg-3 p-2 py-5 bg-light border text-6 shadom-sm mb-2"
                            )
                        ),
                        class_="col-lg-9 m-0 bg-white"
                    ),

                    class_="row"
                )

            ),
            ui.nav("LSTM",
                   
               ui.div(
                        ui.div(                       
                                     
                            ui.div(
                                ui.h4(" Hyper parameters values: ", class_="text-black- text-5"),
                                ui.input_slider( "rate_lstm", 

                                    ui.span("Learning rate", class_="text-5 mb-0"),
                                    min = 1e-3, max = 1, animate = True, value = 1e-2, step = 0.01
                                ),
                                ui.input_slider("units_lstm", 

                                    ui.span("Number of LSTM units", class_="text-5 my-2 text-black-"),
                                    min = 1, max = 500, animate = True, value = 100, step = 1
                                ),
                                ui.input_slider("epoch_lstm", 
                                    ui.span("Number of epochs", class_="text-5 my-2 text-black-"),
                                    min = 1, max = 500, animate = True, value = 10, step = 1
                                ),
                                ui.input_slider("batch_lstm", 
                                    ui.span("Batch size", class_="text-5 my-2 text-black-"),
                                    min = 1, max = 500, animate = True, value = 100, step = 1
                                ) 
                            )
                          
                    ,           
                        class_="col-lg-3 p-4 m-0 bg-light-"
                    ),

                    ui.div(
                        ui.div( {"class": "row"},
                            ui.div(
                                ui.row( {"class": "gx-2"},
                                    ui.column(8,
                                        ui.input_slider(
                                            "choose_model",
                                            ui.h6("Select the time window:", class_="text-5 text-black-"),
                                            min=Date(2000, 1, 1),
                                            max=Date(2022, 12, 1),    
                                            step=1,                            
                                            value=[Date(2016, 1, 1), Date(2016, 5, 1)], drag_range=True,
                                            time_format="%Y-%m-%d"
                                        ),
                                        class_="d-none"
                                    ),
                                    ui.column(6,                                        
                                        ui.input_date_range(
                                            "daterange_calage", "Calage:", start=Date(1965, 1, 1), end=Date(1972, 12, 30), language="fr"
                                        ),
                                    ),
                                    ui.column(6,
                                        ui.input_date_range(
                                            "daterange_validation", "Validation:", start=Date(1972, 12, 31), end=Date(1979, 12, 30), language="fr"
                                        ),
                                    ),
                                ),

                                ui.div(
                                   ui.output_plot("lstm_plot"),
                                # ui.download_button("download_plot", "Download"),
                                 ui.output_plot("corr_obs_lstm"),
                                 ui.output_plot("corr_sim_lstm"),
                                 #   ui.output_plot("corr_obs_sim"),
                               ),

                                class_="col-lg-9 py-2 px-4 bg-white"
                            ),
                            ui.div(
                                ui.div(
                                    ui.row(
                                        ui.column(4, "Criteria"),
                                        ui.column(4, "Qsim"),
                                        ui.column(4, "Qold"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[Q]"),
                                        ui.column(4, "0.28"),
                                        ui.column(4, "0.81"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[sqrt(Q)]"),
                                        ui.column(4, "0.52"),
                                        ui.column(4, "0.63"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "MAE"),
                                        ui.column(4, "0.4"),
                                        ui.column(4, "0.36"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "RMSE"),
                                        ui.column(4, "0.35"),
                                        ui.column(4, "0.45"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "R²"),
                                        ui.column(4, "0.85"),
                                        ui.column(4, "0.64"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                ),  
                                class_="col-lg-3 p-2 py-5 bg-light border text-6 shadom-sm mb-2"
                            )
                        ),
                        class_="col-lg-9 m-0 bg-white"
                    ),

                    class_="row"
                )

            ),
            ui.nav("GRU",
                ui.div(
                        ui.div(                       
                                     
                            ui.div(
                    ui.h4("Hyper parameters values: ", class_="text-black- text-5"),
                    ui.input_slider("rate_gru", 
                        ui.span("Learning rate", class_="text-5 mb-0"),
                        min = 1, max = 500, animate = True, value = 100, step = 1
                    ),
                    ui.input_slider("units_gru", 
                        ui.span("Number of GRU units", class_="text-5 my-2 text-black-"),
                        min = 1, max = 500, animate = True, value = 100, step = 1
                        ),
                    ui.input_slider("epoch_gru", 
                        ui.span("Number of epochs", class_="text-5 my-2 text-black-"),
                        min = 1, max = 500, animate = True, value = 10, step = 1
                                    ),
                                    ui.input_slider(
                                        "batch_gru", 
                                        ui.span("Batch size", class_="text-5 my-2 text-black-"),
                                        min = 1, max = 500, animate = True, value = 100, step = 1
                    )   
                )
                          
                    ,           
                        class_="col-lg-3 p-4 m-0 bg-light-"
                    ),

                    ui.div(
                        ui.div( {"class": "row"},
                            ui.div(
                                ui.row( {"class": "gx-2"},
                                    ui.column(8,
                                        ui.input_slider(
                                            "choose_model",
                                            ui.h6("Select the time window:", class_="text-5 text-black-"),
                                            min=Date(2000, 1, 1),
                                            max=Date(2022, 12, 1),    
                                            step=1,                            
                                            value=[Date(2016, 1, 1), Date(2016, 5, 1)], drag_range=True,
                                            time_format="%Y-%m-%d"
                                        ),
                                        class_="d-none"
                                    ),
                                    ui.column(6,                                        
                                        ui.input_date_range(
                                            "daterange_calage", "Calage:", start=Date(1965, 1, 1), end=Date(1972, 12, 30), language="fr"
                                        ),
                                    ),
                                    ui.column(6,
                                        ui.input_date_range(
                                            "daterange_validation", "Validation:", start=Date(1972, 12, 31), end=Date(1979, 12, 30), language="fr"
                                        ),
                                    ),
                                ),

                                ui.div(
                                   ui.output_plot("gru_plot_second"),
                                # ui.download_button("download_plot", "Download"),
                                 ui.output_plot("corr_obs_gru"),
                                 ui.output_plot("corr_sim_gru"),
                                 #   ui.output_plot("corr_obs_sim"),
                               ),

                                class_="col-lg-9 py-2 px-4 bg-white"
                            ),
                            ui.div(
                                ui.div(
                                    ui.row(
                                        ui.column(4, "Criteria"),
                                        ui.column(4, "Qsim"),
                                        ui.column(4, "Qold"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[Q]"),
                                        ui.column(4, "0.28"),
                                        ui.column(4, "0.81"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[sqrt(Q)]"),
                                        ui.column(4, "0.52"),
                                        ui.column(4, "0.63"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "MAE"),
                                        ui.column(4, "0.4"),
                                        ui.column(4, "0.36"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "RMSE"),
                                        ui.column(4, "0.35"),
                                        ui.column(4, "0.45"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "R²"),
                                        ui.column(4, "0.85"),
                                        ui.column(4, "0.64"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                ),  
                                class_="col-lg-3 p-2 py-5 bg-light border text-6 shadom-sm mb-2"
                            )
                        ),
                        class_="col-lg-9 m-0 bg-white"
                    ),

                    class_="row"
                )

                
            ),
            ui.nav("GR4J",
                   ui.div(
                        ui.div(                       
                                     
                            ui.div(
                                ui.h4("Parameters values: ", class_="text-black- text-5"),
                                ui.input_slider(
                                    "modypmad_x1", 
                                    ui.span("X1 (production store capacity)", class_="text-5 mb-0"),
                                    min = 0, max = 2, animate = True, value = 1.12, step = 0.01, post=" [mm]"
                                ),
                                ui.input_slider(
                                    "modypmad_x2", 
                                    ui.span("X2 (groundwater exchange coeff.", class_="text-5 my-2 text-black-"),
                                    min = 0, max = 100, animate = True, value = 35.58, step = 0.01, post=" [mm/d]"
                                ),
                                ui.input_slider(
                                    "modypmad_x3", 
                                    ui.span("X3 (routing store capacity)", class_="text-5 my-2 text-black-"),
                                    min = 0, max = 1, animate = True, value = 0.15, step = 0.01, post=" [mm]"
                                ),
                                ui.input_slider(
                                    "modypmad_x4", 
                                    ui.span("X4 (unit hydrograph time constant)", class_="text-5 my-2 text-black-"),
                                    min = 0, max = 1, animate = True, value = 0.42, step = 0.01, post=" [d]"
                                )   
                            )
                          
                    ,           
                        class_="col-lg-3 p-4 m-0 bg-light-"
                    ),

                    ui.div(
                        ui.div( {"class": "row"},
                            ui.div(
                                ui.row( {"class": "gx-2"},
                                    ui.column(8,
                                        ui.input_slider(
                                            "choose_model",
                                            ui.h6("Select the time window:", class_="text-5 text-black-"),
                                            min=Date(2000, 1, 1),
                                            max=Date(2022, 12, 1),    
                                            step=1,                            
                                            value=[Date(2016, 1, 1), Date(2016, 5, 1)], drag_range=True,
                                            time_format="%Y-%m-%d"
                                        ),
                                        class_="d-none"
                                    ),
                                    ui.column(6,                                        
                                        ui.input_date_range(
                                            "daterange_calage", "Calage:", start=Date(1965, 1, 1), end=Date(1972, 12, 30), language="fr"
                                        ),
                                    ),
                                    ui.column(6,
                                        ui.input_date_range(
                                            "daterange_validation", "Validation:", start=Date(1972, 12, 31), end=Date(1979, 12, 30), language="fr"
                                        ),
                                    ),
                                ),

                                ui.div(
                                   ui.output_plot("gr4j_plot"),
                                #    ui.download_button("download_plot", "Download"),
                                   ui.output_plot("corr_obs_gr4j"),
                                   ui.output_plot("corr_sim_gr4j"),
                                 #   ui.output_plot("corr_obs_sim"),
                               ),


                                class_="col-lg-9 py-2 px-4 bg-white"
                            ),
                            ui.div(
                                ui.div(
                                    ui.row(
                                        ui.column(4, "Criteria"),
                                        ui.column(4, "Qsim"),
                                        ui.column(4, "Qold"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[Q]"),
                                        ui.column(4, "0.28"),
                                        ui.column(4, "0.81"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "NSE[sqrt(Q)]"),
                                        ui.column(4, "0.52"),
                                        ui.column(4, "0.63"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "MAE"),
                                        ui.column(4, "0.4"),
                                        ui.column(4, "0.36"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "RMSE"),
                                        ui.column(4, "0.35"),
                                        ui.column(4, "0.45"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                    ui.row(
                                        ui.column(4, "R²"),
                                        ui.column(4, "0.85"),
                                        ui.column(4, "0.64"),
                                    ),
                                    ui.hr({"class":"p-0 my-2"}),
                                ),  
                                class_="col-lg-3 p-2 py-5 bg-light border text-6 shadom-sm mb-2"
                            )
                        ),
                        class_="col-lg-9 m-0 bg-white"
                    ),

                    class_="row"
                )

                   
            ),

            ui.nav("Statistical analysis", 
                 ui.div(
                    ui.navset_pill_list(
                        ui.nav("Hydro-meteorological data", 
                            ui.div(
                                ui.row(
                                    ui.div({"class": "col-md-6 mt-4"}, 
                                        ui.output_plot("precipitations_2"),
                                    ),
                                    ui.div({"class": "col-md-6 mt-4"}, 
                                        ui.output_plot("etp"),
                                    ),
                                    ui.div({"class": "mt-4"}, 
                                        ui.output_plot("debit"),
                                    ),
                                ),
                            ),
                        ),
                        
                        ui.nav("Inter-annual variability", 
                            ui.div(
                                ui.row(
                                    ui.div({"class": "col-md-6 mt-4"}, 
                                        ui.output_plot("precipitations"),
                                    ),
                                    ui.div({"class": "col-md-6 mt-4"}, 
                                        ui.output_plot("ecoulement"),
                                    ),
                                    ui.div({"class": "mt-4"}, 
                                        ui.output_plot("precipitations_ecoulement"),
                                    ),
                                ),
                            ),
                        ),
                       
                        ui.nav("Linear regression", 
                            ui.div(
                                ui.row(
                                    ui.div({"class": "mt-4"}, 
                                        ui.output_plot("debit_corr"),
                                    ),
                                ),
                            ),
                        ),
                        ui.nav("Statistics", 
                            ui.div(
                                ui.row(
                                    ui.div({"class": "my-4"}, 
                                        ui.output_plot("stats_desc"),
                                    ),
                                    ui.div({"class": "mt-5"}, 
                                        ui.div(ui.output_ui("contents")),
                                    ),
                                    ui.div({"class": "my-5"},
                                        ui.output_table("data_table")
                                    )
                                ),
                            ),
                        ),
                    ),
                    
                    class_="row"
                )

            ),
            ui.nav("Hyper Parameter Tuning", 
                ui.div(
                     ui.h2("Bayesian optimization", class_="text-center text-primary-"),
                     class_="bg-light- py-4"
                   
                ),
                ui.div(
                      ui.input_select("choose_opt", "Recurrent Neural Network", {"LSTM": "LSTM", "GRU": "GRU"}),
                      ui.h4(" Hyper parameters values: ", class_="text-black- text-5"),
                      ui.div(
                        ui.input_numeric("rate_inf", "Learning rate between", value=1e-3),
                        ui.input_numeric("rate_sup", "", value=1),
                        ui.input_numeric("num_unit_inf", "Number of Units between", value=1),
                        ui.input_numeric("num_unit_sup", "", value=300),
                        ui.input_numeric("num_epoch_inf", "Number of epochs between", value=1),
                        ui.input_numeric("num_epoch_sup", "", value=300),
                        ui.input_numeric("num_batch_inf", "Batch size between", value=1),
                        ui.input_numeric("num_batch_sup", "", value=300),
                        #ui.hr(),
                        ui.div(
                        ui.h4("Hyper-parameter Tuning:", class_="text-black- text-5"),
                                                  
                        ui.input_action_button(
                                "run", "Run optimization", class_="btn-primary bg-primary- w-25"
                  
                      )
                    )
                )
       

                )
                
            ),
            ui.nav_spacer(),

            ui.nav("About", 
                ui.div(
                    ui.h2("A propos de ModHyLog", class_="text-center text-primary-"),
                    class_="bg-light- py-4"
                ),
                ui.row(
                    ui.column(6, 
                        ui.div()
                    ),
                )
            ),
            ui.nav("Help", 
                ui.div(
                    ui.p(
                        "Aide"
                    ),
                    class_="col-md-6 mx-auto bg-primary- rounded shadow p-5"
                )
            ),
            ui.nav("Contact", 
                ui.div(
                    ui.row(
                        ui.column(6, "Email : ", class_="text-white- text-5"),
                        ui.column(6, 
                            ui.a("contact@modhylog.com", href="mailto:contact@modhylog.com", class_="text-white- text-5")
                        ),
                    ),
                    ui.row(
                        ui.column(6, "Téléphone : ", class_="text-white- text-5"),
                        ui.column(6, 
                            ui.a("+229 96 00 00 00", href="tel:+229 96 00 00 00", class_="text-white- text-5"), 
                        ),
                        class_="mt-4"
                    ),
                    ui.row(
                        ui.column(6, "Whatsapp : ", class_="text-white- text-5"),
                        ui.column(6, 
                            ui.a("+229 96 00 00 00", href="https://wa.me/22996000000", class_="text-white- text-5"), 
                        ),
                        class_="mt-4"
                    ),
                    ui.row(
                        ui.column(6, "Linkedin : ", class_="text-white- text-5"),
                        ui.column(6, 
                            ui.a("https://www.linkedin.com/", href="https://www.linkedin.com/", class_="text-white- text-5"), 
                        ),
                        class_="mt-4"
                    ),
                    ui.row(
                        ui.column(6, "Adresse : ", class_="text-white- text-5"),
                        ui.column(6, 
                            "Université d'Abomey-Calavi/IFRI",
                            class_="text-white- text-5" 
                        ),
                        class_="mt-4"
                    ),
                    class_="col-md-6 mx-auto bg-primary- rounded shadow p-5 my-5"
                )
            ),
        ),


    ),
)

def nse(q_obs,q_sim):
    return 1-  np.sum((q_obs-q_sim)**2) / np.sum((q_obs - np.mean(q_obs))**2)

def modhypma_stochastic(x, data):

    m = x[0]
    l = x[1]
    tx = x[2]
    p2 = x[3]


    #data = pd.read_csv(file)   
    P, ETP, q_obs = np.array(data['Prec']), np.array(data['ETP']), np.array(data['Debit'])
                 # Computation of effective rainfall
    P = [float(x) for x in P]
    q = [max(0,x) for x in P - ETP]
    a =1
    b= int(0.5*len(q_obs))
    diff_Q =  q_obs[a-1:b-1]-q_obs[a:b]
            # Computation of soil's states
    size = b-a+1
    X = [0.0]
            
            # state oil 
    for i in range(1,size):
        try:
            if q[i] == 0:
                X.append(X[i-1] - (m/l) * X[i-1])
            else:
                X.append(X[i-1] + (m/l) * q[i]**(2*m - p2))
        except:
            print("Error of soil state computation : Bad dataset !")

            # Limitation of the state of soil by introduce TX
    for x in X:
        if x < tx:
            x = 0
            
    Q= np.zeros((size,1))
    q_obs= np.array(q_obs)
    X= np.array(X)
    q= np.array(q)
    Q[0]= q_obs[a]
    Q[1]= q_obs[a+1]

    for k in range(1,size-1):
        Q[k+1] = Q[k] + (1/l)*X[k+1]*q[k] - (m/l)*(Q[k])**(2*m - 1)

        if Q[k+1] < 0:
            Q[k+1] = 0
            
    Y=  np.transpose((1/l)*(X[1:b]))*q[0:b-1] - (m/l)*(Q[0:b-1])**(2*m - 1)
    e= diff_Q -Y
    R =  np.mean(e)
    G=  np.std(e)

            # deuxième partie
    a=  int(0.5*len(q_obs))+ 1
    b=  len(q_obs)
    qe= q[a-1:b]
    size = b-a+1
    X = [0.0] 
          
                        
            # state oil 
    for i in range(1,size):
        try:
            if qe[i] == 0:
                X.append(X[i-1] - (m/l) * X[i-1])
            else:
                X.append(X[i-1] + (m/l) * qe[i]**(2*m - p2))
        except:
            print("Error of soil state computation : Bad dataset !")

            # Limitation of the state of soil by introduce TX
    for x in X:
        if x < tx:
            x = 0
            
    Q= np.zeros((size,1))
    q_obs= np.array(q_obs)
    X= np.array(X)
    q= np.array(qe)
    Q[0]= q_obs[a]
    Q[1]= q_obs[a+1]

    for k in range(1,size-1):
        Q[k+1] = Q[k] + (1/l)*X[k+1]*q[k] - (m/l)*(Q[k])**(2*m - 1)

        if Q[k+1] < 0:
            Q[k+1] = 0
            
            # Computation of q_sim
    q_sim = np.zeros((size,1))
    q_sim[0] = q_obs[a]
    q_sim[1] = q_obs[a+1]
    seuil = 0
    
    while seuil <0.85:
        for j in range(2,size):
                f=  (1/l)*X[j]*qe[j-1] - (m/l)*q_sim[j-1]**(2*m-1) +R
                dB =np.random.randn(1)
                q_sim[j] = q_sim[j-1] +  f+ G*dB
                if q_sim[j] < 0:
                    q_sim[j] = 0 
                
        Nash_Total2 = nse(q_obs[a:b], q_sim)
        if seuil < Nash_Total2:
            seuil = Nash_Total2 
            BestQstoch= q_sim
    
         
    performance_validation =  criteria.performance_calage(q_obs[a-1:b],BestQstoch)

    return q_obs, q_sim,performance_validation



def server(input, output, session):
        
    @output 
    @render.plot(alt="A histogram")
    def gr4j_plot():
        data = pd.read_excel('modhypmahydro/csv/PEQ.xlsx')
        data.columns = ['Dates','Prec','ETP','Debit']

        data['Dates'] = pd.to_datetime(data['Dates'], errors='coerce')
        data['Dates'] = data['Dates'].dt.strftime('%d/%m/%Y')
        
        

        cal_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
        val_date =  val_data['Dates']
        dates_obs = val_date[:len(val_data['Debit'])]
        params = {'X1': 235.32, 'X2': -19.99, 'X3': 199.99, 'X4': 2.73}

        simulated_flow,mae, mse, rmse, nse, r2 = gr4j_debit(cal_data, val_data, params)

        # Plot the observations and predictions
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dates_obs, val_data['Debit'], label='observations')
        ax.plot(dates_obs, simulated_flow, label='GR4J predictions')
        ax.set_xlabel('Dates', fontsize=16)
        ax.set_ylabel('Discharge ($m^3/s$)', fontsize=16)
        ax.legend()
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of displayed ticks

        plt.tight_layout()
        return  fig

    @output
    @render.plot(alt=" A histogram ")
    def corr_sim_gr4j():
        if input.file() is None:    
            data = data = pd.read_excel('modhypmahydro/csv/PEQ.xlsx')
            data.columns = ['Dates', 'Prec', 'ETP', 'Debit']
        else:
            f: list[FileInfo] = input.file()
            data = pd.read_csv(f[0]["datapath"], header=0)
            data.columns = ['Dates', 'Prec', 'ETP', 'Debit']

        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)

        cal_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
        
                
        cal_date = cal_data['Dates']
        val_date = val_data['Dates']
        

        params = {'X1': 235.32, 'X2': -19.99, 'X3': 199.99, 'X4': 2.73}

        simulated_flow,mae, mse, rmse, nse, r2 = gr4j_debit(cal_data, val_data, params)

        fig, ax = plt.subplots(figsize=(8, 6))
        X = simulated_flow[1:]
        Y = simulated_flow[:-1]

        ax.scatter(X, Y,label='(Q_sim_t,Q_sim_t-1)')
        ax.set_xlabel("Débits à la date t ($m^3/s$)")
        ax.set_ylabel("Débits à la date t-1 ($m^3/s$)")
        ax.legend(fontsize=10)

        d = linregress(X, Y)
        Z = np.linspace(0, max(X))
        eqt = f"y = {round(d.slope, 4)}x + {round(d.intercept, 4)}"
        ax.plot(Z, d.slope * Z + d.intercept, label=eqt, color='red')
        ax.legend(fontsize=10)

        plt.tight_layout()

        return fig
    
    @output
    @render.plot(alt=" A histogram ")
    def corr_obs_gr4j():


        if input.file() is None:    
            data = pd.read_excel('modhypmahydro/csv/PEQ.xlsx') 
            data.columns = ['Dates','Prec','ETP','Debit']
        else:
            f: list[FileInfo] = input.file()
            data = pd.read_csv(f[0]["datapath"], header=0)
            data.columns = ['Dates','Prec','ETP','Debit']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        debits = data['Debit']
        x = debits.values
        X = x[1:]
        Y = x[:-1]

        fig, ax = plt.subplots(figsize=(8, 6))  # Set the figure size directly here using `figsize`
        ax.scatter(X, Y, label='(Q_obs_t, Q_obs_t-1)')
        ax.set_xlabel("Débits à la date t ($m^3/s$)")  # Use `ax.set_xlabel` instead of `plt.xlabel`
        ax.set_ylabel("Débits à la date t-1 ($m^3/s$)")  # Use `ax.set_ylabel` instead of `plt.ylabel`
        ax.legend(fontsize=10)

        d = linregress(X, Y)
        Z = np.linspace(0, max(X))
        eqt = f"y = {round(d.slope, 4)}x + {round(d.intercept, 4)}"
        ax.plot(Z, d.slope * Z + d.intercept, label=eqt, color='red')
        ax.legend(fontsize=10)

        plt.tight_layout()  # Add this line to improve the layout of the plot

        return fig

    @output
    @render.plot(alt="A histogram")
    def stoch_plot():
        if input.file() is None:    
            data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
            data.columns = ['Dates','Prec','ETP','Debit']
        else:
            f: list[FileInfo] = input.file()
            data = pd.read_csv(f[0]["datapath"], header=0)
            data.columns = ['Dates','Prec','ETP','Debit']

        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)


        params= [0.9704731595960552,10.073946467755349, 0.19391146970833906,0.6694441401961488]
        q_obs, q_sim,performance_validation = modhypma_stochastic(params, data)

        #print("Performance validation", performance_validation)
    
        time1 =  date.toordinal(date(daterange_validation[0].year,daterange_validation[0].month,daterange_validation[0].day))
        time2 =  date.toordinal(date(daterange_validation[1].year,daterange_validation[1].month,daterange_validation[1].day))

        time = np.arange(time1,time2+1)        
        
        # Graphiques d'ensemble 
        x = q_sim
        abs = np.arange(len(x[0]))
        p =data['Prec'].iloc[0:len(x[0])]
        fig, ax = plt.subplots()
        ax.bar(abs,p,color=[0,0.1,0.2] )
        color = 'tab:red'
        ax.xaxis_date()
        ax.invert_yaxis()
        ax.set_ylabel('Y1-axis', color = color) 
        ax2 = ax.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Y2-axis', color = color)  
        ax2.plot(abs,x[1])
        ax2.plot(abs,x[0])

        ax.set_xlabel("Date in day")
        ax.set_ylabel("Discharge")
        

    @output
    @render.plot(alt="A histogram")
       
    def plot():  
        #data = None 
        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise

         
        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)

        
        debits = get_debits(input.modypmad_x1(), input.modypmad_x2(), input.modypmad_x3(), input.modypmad_x4(), data, cal_start, cal_end, val_start, val_end)
    
    
        time1 =  date.toordinal(date(daterange_validation[0].year,daterange_validation[0].month,daterange_validation[0].day))
        time2 =  date.toordinal(date(daterange_validation[1].year,daterange_validation[1].month,daterange_validation[1].day))

        time = np.arange(time1,time2+1)        
        
        # Graphiques d'ensemble 
        x = debits
        abs = np.arange(len(x[0]))
        p =data['Prec'].iloc[0:len(x[0])]
        fig, ax = plt.subplots()
        ax.bar(abs,p,color=[0,0.1,0.2] )
        color = 'tab:red'
        ax.xaxis_date()
        ax.invert_yaxis()
        ax.set_ylabel('Y1-axis', color = color) 
        ax2 = ax.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Y2-axis', color = color)  
        ax2.plot(abs,x[1])
        ax2.plot(abs,x[0])

        ax.set_xlabel("Date in day")
        ax.set_ylabel("Discharge")
        
        return fig
    
    @output
    @render.plot(alt="A histogram")
    def corr_obs():
        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise
        
        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)
        
    

        debits = get_debits(input.modypmad_x1(), input.modypmad_x2(), input.modypmad_x3(), input.modypmad_x4(), data, cal_start, cal_end, val_start, val_end)
        
        x = debits
        abs = np.arange(len(debits[0]))
        fig, ax = plt.subplots()

        X = np.array(x[0][1:len(x[0])-1])
        Y = np.array(x[0][0:len(x[0])-2])
        ax.scatter(X,Y, label='(Q_obs_t,Q_obs_t-1)')
        ax.set_xlabel("Débits à la date t ($m^3/s$)")
        ax.set_ylabel("Débits à la date t-1 ($m^3/s$)")

        ax.legend(fontsize=10)
    
        d = linregress(X,Y)
        Z = np.linspace(0, max(x[0]))
        eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
        ax.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
        ax.legend(fontsize=10)

        
        
        return fig
    
    @output
    @render.plot(alt=" A histogram")
    def lstm_plot():
        global inv_yhat_lstm, inv_y_lstm

        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise
        
        
        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)


        #print("Début lstm modele")
        #cal_data, val_data = split_cal_val(data, cal_start , cal_end , val_start, val_end)
        cal_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
        learning_rate = input.rate_lstm() # Taux d'apprentissage
        num_lstm_units = input.units_lstm() # Nombre d'unités LSTM dans votre modèle
        num_epochs = input.epoch_lstm()  # Nombre d'époques pour l'entraînement
        num_batch_size =input.batch_lstm()  # Taille du lot (batch size) pour l'entraînement
        cal_date =  cal_data['Dates']
        val_date =  val_data['Dates']

        inv_yhat_lstm,inv_y_lstm, R2,nse, rmse, mae = get_lstm_debits(data, cal_data=cal_data,val_data= val_data,learning_rate=learning_rate ,num_lstm_units = num_lstm_units,num_epochs=  num_epochs,num_batch_size= num_batch_size)
        dates_obs = val_date[:-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dates_obs,inv_y_lstm, label='observations')
        ax.plot(dates_obs,inv_yhat_lstm, label='LSTM predictions')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Discharge ($m^3/s$)', fontsize=16)
        ax.legend()
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limiter le nombre de ticks affichés

        plt.tight_layout()
        return fig
    
    

    @output
    @render.plot(alt=" A histogram")
    def gru_plot_second():
        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise
        cal_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)

        cal_date = cal_data['Dates']
        val_date = val_data['Dates']

        """
        learning_rate = input.rate_gru() # Taux d'apprentissage
        num_gru_units = input.units_gru() # Nombre d'unités LSTM dans votre modèle
        num_epochs = input.epoch_gru()  # Nombre d'époques pour l'entraînement
        num_batch_size =input.batch_gru() 
        """
        learning_rate = 1e-2
        num_gru_units = 100
        num_epochs=10
        num_batch_size= 100

        inv_yhat, inv_y, R2, nse, rmse, mae = gru_debits(data, cal_data, val_data, learning_rate, num_gru_units, num_epochs, num_batch_size)
        dates_obs = val_date[:-1]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dates_obs, inv_y, label='observations')
        ax.plot(dates_obs, inv_yhat, label='GRU predictions')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Discharge ($m^3/s$)', fontsize=16)
        ax.legend()
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limiter le nombre de ticks affichés

        plt.tight_layout()
        return fig


    


    @output
    @render.plot(alt=" A histogram ")
    def corr_obs_lstm():


        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise
        
        fig, ax = plt.subplots(figsize=(8, 6))
        debits = data['Debit']
        x = debits.values
        X = x[1:]
        Y = x[:-1]

        fig, ax = plt.subplots(figsize=(8, 6))  # Set the figure size directly here using `figsize`
        ax.scatter(X, Y, label='(Q_obs_t, Q_obs_t-1)')
        ax.set_xlabel("Débits à la date t ($m^3/s$)")  # Use `ax.set_xlabel` instead of `plt.xlabel`
        ax.set_ylabel("Débits à la date t-1 ($m^3/s$)")  # Use `ax.set_ylabel` instead of `plt.ylabel`
        ax.legend(fontsize=10)

        d = linregress(X, Y)
        Z = np.linspace(0, max(X))
        eqt = f"y = {round(d.slope, 4)}x + {round(d.intercept, 4)}"
        ax.plot(Z, d.slope * Z + d.intercept, label=eqt, color='red')
        ax.legend(fontsize=10)

        plt.tight_layout()  # Add this line to improve the layout of the plot

        return fig

    
    @output
    @render.plot(alt=" A histogram ")
    def corr_obs_gru():


        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise
        
        fig, ax = plt.subplots(figsize=(8, 6))
        debits = data['Debit']
        x = debits.values
        X = x[1:]
        Y = x[:-1]

        fig, ax = plt.subplots(figsize=(8, 6))  # Set the figure size directly here using `figsize`
        ax.scatter(X, Y, label='(Q_obs_t, Q_obs_t-1)')
        ax.set_xlabel("Débits à la date t ($m^3/s$)")  # Use `ax.set_xlabel` instead of `plt.xlabel`
        ax.set_ylabel("Débits à la date t-1 ($m^3/s$)")  # Use `ax.set_ylabel` instead of `plt.ylabel`
        ax.legend(fontsize=10)

        d = linregress(X, Y)
        Z = np.linspace(0, max(X))
        eqt = f"y = {round(d.slope, 4)}x + {round(d.intercept, 4)}"
        ax.plot(Z, d.slope * Z + d.intercept, label=eqt, color='red')
        ax.legend(fontsize=10)

        plt.tight_layout()  # Add this line to improve the layout of the plot

        return fig
    
        
    @output
    @render.plot(alt=" A histogram ")
    def corr_sim_lstm():
        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise

        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)

        cal_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
        learning_rate = input.rate_lstm() # Taux d'apprentissage
        num_lstm_units = input.units_lstm() # Nombre d'unités LSTM dans votre modèle
        num_epochs = input.epoch_lstm()  # Nombre d'époques pour l'entraînement
        num_batch_size =input.batch_lstm() 
        cal_date = cal_data['Dates']
        val_date = val_data['Dates']

        inv_yhat_lstm, inv_y_lstm, R2, nse, rmse, mae = get_lstm_debits(data, cal_data=cal_data, val_data=val_data,
                                                                        learning_rate=learning_rate,
                                                                        num_lstm_units=num_lstm_units,
                                                                        num_epochs=num_epochs,
                                                                        num_batch_size=num_batch_size)

        fig, ax = plt.subplots(figsize=(8, 6))
        X = inv_yhat_lstm[1:]
        Y = inv_yhat_lstm[:-1]

        ax.scatter(X, Y,label='(Q_sim_t,Q_sim_t-1)')
        ax.set_xlabel("Débits à la date t ($m^3/s$)")
        ax.set_ylabel("Débits à la date t-1 ($m^3/s$)")
        ax.legend(fontsize=10)

        d = linregress(X, Y)
        Z = np.linspace(0, max(X))
        eqt = f"y = {round(d.slope, 4)}x + {round(d.intercept, 4)}"
        ax.plot(Z, d.slope * Z + d.intercept, label=eqt, color='red')
        ax.legend(fontsize=10)

        plt.tight_layout()

        return fig

    @output
    @render.plot(alt=" A histogram ")
    def corr_sim_gru():
        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise

        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)

        cal_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
        """
        learning_rate = input.rate_gru() # Taux d'apprentissage
        num_gru_units = input.units_gru() # Nombre d'unités LSTM dans votre modèle
        num_epochs = input.epoch_gru()  # Nombre d'époques pour l'entraînement
        num_batch_size =input.batch_gru() 
        """

        learning_rate = 1e-2
        num_gru_units = 100
        num_epochs=10
        num_batch_size= 100
                
        cal_date = cal_data['Dates']
        val_date = val_data['Dates']
        

        inv_yhat, inv_y, R2, nse, rmse, mae = gru_debits(data, cal_data, val_data, learning_rate, num_gru_units, num_epochs, num_batch_size)
       

        fig, ax = plt.subplots(figsize=(8, 6))
        X = inv_yhat[1:]
        Y = inv_y[:-1]

        ax.scatter(X, Y,label='(Q_sim_t,Q_sim_t-1)')
        ax.set_xlabel("Débits à la date t ($m^3/s$)")
        ax.set_ylabel("Débits à la date t-1 ($m^3/s$)")
        ax.legend(fontsize=10)

        d = linregress(X, Y)
        Z = np.linspace(0, max(X))
        eqt = f"y = {round(d.slope, 4)}x + {round(d.intercept, 4)}"
        ax.plot(Z, d.slope * Z + d.intercept, label=eqt, color='red')
        ax.legend(fontsize=10)

        plt.tight_layout()

        return fig


    @output
    @render.plot(alt="A histogram")
    def corr_sim():
        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise
        
        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)

        

        debits = get_debits(input.modypmad_x1(), input.modypmad_x2(), input.modypmad_x3(), input.modypmad_x4(), data, cal_start, cal_end, val_start, val_end)
        metrics = {'mae': mean_absolute_error(debits[0],debits[1]), 'mse':root_mean_squared_error(debits[0],debits[1]), 'rmse':mean_absolute_error(debits[0],debits[1]), 'nse':nash_sutcliffe_efficiency(debits[0],debits[1]), 'r2':coefficient_of_determination(debits[0],debits[1])}
            
        x = debits
        fig, ax = plt.subplots()

        X = np.array(x[1][1:len(x[1])-1])
        Y = np.array(x[1][0:len(x[1])-2])
        ax.scatter(X,Y, label='(Q_sim_t,Q_sim_t-1)')
        ax.set_xlabel("Débits à la date t ($m^3/s$)")
        ax.set_ylabel("Débits à la date t-1 ($m^3/s$)")
        ax.legend(fontsize=10)
        
        d = linregress(X,Y)
        print(d)
        Z = np.linspace(0, max(x[1]))
        eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
        ax.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
        ax.legend(fontsize=10)

        

        return fig
    
    @output
    @render.plot(alt="A histogram")
    def corr_obs_sim():
        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise
        
        
        daterange_calage = list(input.daterange_calage())
        daterange_validation = list(input.daterange_validation())

        cal_start = str(daterange_calage[0].day).zfill(2)+"/"+str(daterange_calage[0].month).zfill(2)+"/"+str(daterange_calage[0].year)
        cal_end =  str(daterange_calage[1].day).zfill(2)+"/"+str(daterange_calage[1].month).zfill(2)+"/"+str(daterange_calage[1].year)
        val_start = str(daterange_validation[0].day).zfill(2)+"/"+str(daterange_validation[0].month).zfill(2)+"/"+str(daterange_validation[0].year)
        val_end =  str(daterange_validation[1].day).zfill(2)+"/"+str(daterange_validation[1].month).zfill(2)+"/"+str(daterange_validation[1].year)

        

        debits = get_debits(input.modypmad_x1(), input.modypmad_x2(), input.modypmad_x3(), input.modypmad_x4(), data, cal_start, cal_end, val_start, val_end)
        metrics = {'mae': mean_absolute_error(debits[0],debits[1]), 'mse':root_mean_squared_error(debits[0],debits[1]), 'rmse':mean_absolute_error(debits[0],debits[1]), 'nse':nash_sutcliffe_efficiency(debits[0],debits[1]), 'r2':coefficient_of_determination(debits[0],debits[1])}
    
        x = debits
        fig, ax = plt.subplots()

        X = np.array(x[0])
        Y = np.array(x[1])
        ax.scatter(X,Y, label='(Q_obs,Q_sim)')
        ax.set_xlabel("Débits observés ($m^3/s$)")
        ax.set_ylabel("Débits simulés ($m^3/s$)")
        ax.legend(fontsize=10)
        
        d = linregress(X,Y)
        print(d)
        Z = np.linspace(0, max(max(x[1]), max(x[0])))
        eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
        ax.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
        ax.legend(fontsize=10)

        

        return fig

    
    @output
    @render.ui
    def contents():
        if input.file() is None:   
            selected_dataset = input.select()
            print("Selected dataset:", selected_dataset)

            if selected_dataset =='1': 
               
                data = pd.read_csv('modhypmahydro/csv/beterou_cal_val.csv') 
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Beterou data loaded successfully!')
               
            elif selected_dataset == '2':
                
                data = pd.read_csv('modhypmahydro/csv/save.csv')
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Save data loaded successfully!')
                
        else:
            f: list[FileInfo] = input.file()
            try:
                data = pd.read_csv(f[0]["datapath"], header=0)
                data.columns = ['Dates','Prec','ETP','Debit']
                print('Custom data loaded successfully!')
            except Exception as e:
                print(f'Error loading custom data: {e}')
                raise

        df1 = data

        data['Dates']= pd.to_datetime(data.Dates, infer_datetime_format=True)
        data['year'] = data['Dates'].dt.year
        precip_sum = data.groupby('year')['Prec'].sum()
        debit_sum = data.groupby('year')['Debit'].sum()
        X = (precip_sum-np.mean(precip_sum))/np.std(precip_sum)
        Y = (debit_sum-np.mean(debit_sum))/np.std(debit_sum)

        df = data
        df.set_index(['Dates'], inplace=True)

        @output
        @render.plot(alt="Précipitations")
        def precipitations():        
            fig, ax = plt.subplots() # plt.figure(figsize = (15, 7))
            ax.bar(X.index,X.values)
            ax.set_ylabel('Anormalies Standardisées')
            ax.set_xlabel('Années')
            ax.set_title('Précipitations')

            return fig

        @output
        @render.plot(alt="Ecoulement")
        def ecoulement():        
            fig, ax = plt.subplots() # plt.figure(figsize = (15, 7))

            ax.bar(Y.index,Y.values,color= 'g')
            ax.set_ylabel('Anormalies Standardisées')
            ax.set_xlabel('Années')
            ax.set_title('Ecoulement')

            return fig

        @output
        @render.plot(alt="Précipitations Ecoulement")
        def precipitations_ecoulement():        
            fig, ax = plt.subplots() # plt.figure(figsize = (15, 7))
            
            width = 0.25

            ax.bar(X.index, X.values, color = 'b',
                width = width, edgecolor = 'black',
                label='Précipitations')
            ax.bar(X.index + width, Y.values, color = 'g',
                width = width, edgecolor = 'black',
                label='Discharge')
            ax.set_ylabel('Anormalies Standardisées')
            ax.set_xlabel('Year')
            ax.set_title('Rainfall Rainoff')

            return fig

        @output
        @render.plot(alt="Précipitations")
        def precipitations_2():        
            fig, ax = plt.subplots() # plt.figure(figsize = (10, 5))

            df['Prec'].plot()
            ax.set_title('Précipitations')

            return fig


        @output
        @render.plot(alt="ETP")
        def etp():        
            fig, ax = plt.subplots() # plt.figure(figsize = (10, 5))
            
            df['ETP'].plot(color='green')
            ax.set_title('ETP')

            return fig

        @output
        @render.plot(alt="Débit")
        def debit():        
            fig, ax = plt.subplots() # plt.figure(figsize = (10, 5))
            
            df['Debit'].plot(color='m')
            ax.set_title('Débit')

            return fig

        @output
        @render.plot(alt="Débit corr")
        def debit_corr():        
            fig, ax = plt.subplots() # plt.figure(figsize = (10, 5))

            Q_t = data['Debit'].iloc[1:data.shape[0]-1]
            Q_t1 =data['Debit'].iloc[2:data.shape[0]]
            ax.scatter(Q_t1,Q_t)
            ax.set_xlabel("Qobs à t-1")
            ax.set_ylabel("Qobs à t")
            ax.legend(title='(Q_t-1,Q_t)',loc ='upper right')

            return fig

        @output
        @render.plot(alt="Stats Desc")
        def stats_desc():        
            ax = data.isna().sum().sort_values().plot(kind = 'barh', figsize = (9, 10))
            fig, ax = plt.subplots() # plt.figure(figsize = (10, 5))
            
            ax.set_title('Pourcentage de valeur manquantes', fontdict={'size':15})
            for p in ax.patches:
                percentage ='{:,.0f}%'.format((p.get_width()/data.shape[0])*100)
                width, height =p.get_width(),p.get_height()
                x=p.get_x()+width+0.02
                y=p.get_y()+height/2
                ax.annotate(percentage,(x,y))

            return fig

        return ui.HTML(df1.describe().transpose().to_html(classes="table table-striped table-bordered"))    
        # return ui.HTML(df.to_html(classes="table table-striped table-bordered"))

        @output
        @render.table
        def data_table():
            return data.head()


    @output
    @render.ui

    def dyn_ui_ai():
        if input.choose_model_ai() == "LSTM": 
            return  ui.div(
                 ui.h4(" Hyper parameters values: ", class_="text-black- text-5"),
                 ui.input_slider( "rate_lstm", 

                    ui.span("Learning rate", class_="text-5 mb-0"),
                    min = 1e-3, max = 1, animate = True, value = 1e-2, step = 0.01
                 ),
                 ui.input_slider("units_lstm", 

                    ui.span("Number of LSTM units", class_="text-5 my-2 text-black-"),
                    min = 1, max = 500, animate = True, value = 100, step = 1
                 ),
                 ui.input_slider("epoch_lstm", 
                    ui.span("Number of epochs", class_="text-5 my-2 text-black-"),
                    min = 1, max = 500, animate = True, value = 100, step = 1
                 ),
                ui.input_slider("batch_lstm", 
                     ui.span("Batch size", class_="text-5 my-2 text-black-"),
                     min = 1, max = 500, animate = True, value = 100, step = 1
                ) 
            )
        elif input.choose_model_ai() == "GRU": 
            return  ui.div(
                ui.h4("Hyper parameters values: ", class_="text-black- text-5"),
                ui.input_slider("rate_gru", 
                    ui.span("Learning rate", class_="text-5 mb-0"),
                    min = 1, max = 500, animate = True, value = 100, step = 1
                ),
                ui.input_slider("units_gru", 
                    ui.span("Number of GRU units", class_="text-5 my-2 text-black-"),
                    min = 1, max = 500, animate = True, value = 100, step = 1
                    ),
                ui.input_slider("epoch_gru", 
                    ui.span("Number of epochs", class_="text-5 my-2 text-black-"),
                    min = 1, max = 500, animate = True, value = 10, step = 1
                                ),
                                ui.input_slider(
                                    "batch_gru", 
                                    ui.span("Batch size", class_="text-5 my-2 text-black-"),
                                     min = 1, max = 500, animate = True, value = 100, step = 1
                                )   
            )

        elif input.choose_model_hydro() == "ModHyPMAD": 
                            return  ui.div(
                                        ui.h4("Parameters values: ", class_="text-black- text-5"),
                                        ui.input_slider(
                                            "modypmad_x1", 
                                            ui.span("X1 (production store capacity)", class_="text-5 mb-0"),
                                            min = 0, max = 2, animate = True, value = 1.12, step = 0.01, post=" [mm]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_x2", 
                                            ui.span("X2 (groundwater exchange coeff.", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 100, animate = True, value = 35.58, step = 0.01, post=" [mm/d]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_x3", 
                                            ui.span("X3 (routing store capacity)", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 1, animate = True, value = 0.15, step = 0.01, post=" [mm]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_x4", 
                                            ui.span("X4 (unit hydrograph time constant)", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 1, animate = True, value = 0.42, step = 0.01, post=" [d]"
                                        )   
                                    )

        elif input.choose_model_hydro() == "GR4J": 
                            return  ui.div(
                                        ui.h4("Parameters values: ", class_="text-black- text-5"),
                                        ui.input_slider(
                                            "modypmad_m", 
                                            ui.span("X1 (production store capacity)", class_="text-5 mb-0"),
                                            min = 0, max = 2, animate = True, value = 1.12, step = 0.01, post=" [mm]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_l", 
                                            ui.span("X2 (groundwater exchange coeff.", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 100, animate = True, value = 35.58, step = 0.01, post=" [mm/d]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_tx", 
                                            ui.span("X3 (routing store capacity)", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 1, animate = True, value = 0.15, step = 0.01, post=" [mm]"
                                        ),
                                        ui.input_slider(
                                            "modypmad_p2", 
                                            ui.span("X4 (unit hydrograph time constant)", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 1, animate = True, value = 0.42, step = 0.01, post=" [d]"
                                        )  ,
                                        ui.input_slider(
                                            "modypmad_p2", 
                                            ui.span("X5 (unit hydrograph time constant)", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 1, animate = True, value = 0.42, step = 0.01, post=" [d]"
                                        )  ,
                                        ui.input_slider(
                                            "modypmad_p2", 
                                            ui.span("X6 (unit hydrograph time constant)", class_="text-5 my-2 text-black-"),
                                            min = 0, max = 1, animate = True, value = 0.42, step = 0.01, post=" [d]"
                                        )   
                                    )

           

app = App(app_ui, server, debug=True)


#shiny run --reload