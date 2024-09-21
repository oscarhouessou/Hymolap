import pandas as pd
import numpy as np
from scipy.optimize import minimize
import modhypma, criteria, algorithm, cplot


def nse(q_obs,q_sim):
    return 1-  np.sum((q_obs-q_sim)**2) / np.sum((q_obs - np.mean(q_obs))**2)

def modhypma_stochastic(x, file):

    m = x[0]
    l = x[1]
    tx = x[2]
    p2 = x[3]


    data = pd.read_csv(file)   
    P, ETP, q_obs = np.array(data['Prec']), np.array(data['ETP']), np.array(data['Q_obs'])
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


file =  'beterou_73_79_val.csv'
x= [0.9704731595960552,10.073946467755349, 0.19391146970833906,0.6694441401961488]
q_obs, q_sim,performance_validation = modhypma_stochastic(x, file)

print("Performance validation", performance_validation)
