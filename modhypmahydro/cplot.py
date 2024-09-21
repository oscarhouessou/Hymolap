import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot(x):
    plt.clf()
    abs = np.arange(len(x[0]))
    plt.plot(abs, x[0], label="Débits observés en ($m^3/s$)")
    plt.plot(abs, x[1], label="Débits simulés en ($m^3/s$)")
    plt.xlabel("Dates  en jour")
    plt.ylabel("Débits")
    plt.legend(fontsize=10)
    plt.show()

def corr_obs(x):
    X = np.array(x[0][1:len(x[0])-1])
    Y = np.array(x[0][0:len(x[0])-2])
    plt.scatter(X,Y, label='(Q_obs_t,Q_obs_t-1)')
    plt.xlabel("Débits à la date t ($m^3/s$)")
    plt.ylabel("Débits à la date t-1 ($m^3/s$)")

    plt.legend(fontsize=10)
    
    d = linregress(X,Y)
    print(d)
    Z = np.linspace(0, max(x[0]))
    eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
    plt.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
    plt.legend(fontsize=10)
    plt.show()

def corr_obs_sim(x):
    X = np.array(x[0])
    Y = np.array(x[1])
    plt.scatter(X,Y, label='(Q_obs,Q_sim)')
    plt.xlabel("Débits observés ($m^3/s$)")
    plt.ylabel("Débits simulés ($m^3/s$)")
    plt.legend(fontsize=10)
    
    d = linregress(X,Y)
    print(d)
    Z = np.linspace(0, max(max(x[1]), max(x[0])))
    eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
    plt.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
    plt.legend(fontsize=10)
    plt.show()

def corr_sim(x):
    X = np.array(x[1][1:len(x[1])-1])
    Y = np.array(x[1][0:len(x[1])-2])
    plt.scatter(X,Y, label='(Q_sim_t,Q_sim_t-1)')
    plt.xlabel("Débits à la date t ($m^3/s$)")
    plt.ylabel("Débits à la date t-1 ($m^3/s$)")
    plt.legend(fontsize=10)
    
    d = linregress(X,Y)
    print(d)
    Z = np.linspace(0, max(x[1]))
    eqt = 'y = ' + str(round(d[0], 4)) + 'x + ' + str(round(d[1], 4))
    plt.plot(Z,d[0]*Z + d[1], label=eqt, color='red')
    plt.legend(fontsize=10)
    plt.show()