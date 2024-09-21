import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def nse(q_obs, q_sim):
    return np.sum((q_obs-q_sim)**2) / np.sum((q_obs - np.mean(q_obs))**2)

def rmse(q_obs, q_sim):
    return np.sqrt(mean_squared_error(q_obs, q_sim))

def mae(q_obs, q_sim):
    return mean_absolute_error(q_obs, q_sim)

def r2(q_obs, q_sim):
    return 1 - r2_score(q_obs, q_sim)

def pbias(q_obs, q_sim):
    return np.fabs(np.sum(q_obs - q_sim) * 100 / np.sum(q_obs))

def rsr(q_obs, q_sim): 
    return np.sqrt(np.sum((q_obs - q_sim)**2)) / np.sqrt( np.sum((q_obs - np.mean(q_obs))**2 ))

def bilan(q_obs, q_sim):
    return np.sum(q_sim) / np.sum(q_obs)

def ai(q_obs, q_sim):
    return np.sum((q_obs - q_sim)**2) / np.sum((np.fabs(np.array(q_sim)-np.mean(q_obs))+np.fabs(q_obs-np.mean(q_obs)))**2)

def nash_log(q_obs, q_sim):
    q_obs = np.array(q_obs)
    q_sim = np.array(q_sim)
    return np.sum((np.sqrt(q_obs + 0.01)-np.sqrt(q_sim + 0.01))**2) / np.sum((np.sqrt(q_obs + 0.01)- np.mean(np.sqrt(q_obs + 0.01)))**2)

def clabourMoore(q_obs, q_sim):
    return np.sum(np.fabs(q_obs - q_sim)) / np.sum(q_obs)

def combine(q_obs, q_sim):
    return nse(q_obs, q_sim) * rmse(q_obs, q_sim)

def bias(q_obs, q_sim):
    return pbias(q_obs, q_sim) / 100

def combine_nse_rmse(q_obs, q_sim):
    return (1-nse(q_obs, q_sim)) * rmse(q_obs, q_sim)

def tri_combine(q_obs, q_sim):
    bias = pbias(q_obs, q_sim) / 100
    from math import fabs
    return nse(q_obs, q_sim) * rmse(q_obs, q_sim) * (1 + fabs(bias))

def tri_combine1(q_obs, q_sim):
    bias = pbias(q_obs, q_sim) / 100
    from math import fabs
    return (1-nse(q_obs, q_sim)) * rmse(q_obs, q_sim) * (1 + fabs(bias))

def combine_moore(q_obs, q_sim):
    return nse(q_obs, q_sim) * clabourMoore(q_obs, q_sim)

def  combine_mae(q_obs, q_sim):
    return nse(q_obs, q_sim) * mae(q_obs, q_sim)

def  combine_moore_mae(q_obs, q_sim):
    return nse(q_obs, q_sim) * clabourMoore(q_obs, q_sim)* mae(q_obs, q_sim)

def combine_moore_mae_rmse(q_obs, q_sim):
    return nse(q_obs, q_sim)  * rmse(q_obs, q_sim) * clabourMoore(q_obs, q_sim) * mae(q_obs, q_sim)

def combine_moore_rmse(q_obs, q_sim):
    return nse(q_obs, q_sim)  * rmse(q_obs, q_sim) * clabourMoore(q_obs, q_sim)

def combine_moore_r2(q_obs, q_sim):
    return nse(q_obs, q_sim)  * r2(q_obs, q_sim)