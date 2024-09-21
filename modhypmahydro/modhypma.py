import pandas as pd
import numpy as np
from scipy.optimize import minimize

class ModHyPMA:
    def __init__(self, x, cal_file, val_file):
        self.m = x[0]
        self.l = x[1]
        self.tx = x[2]
        self.p2 = x[3]
        self.c_file = cal_file
        self.v_file = val_file
        self.file = self.c_file
        self.fct = None

    def loadCsv(self, file):
        """ Read the formatted CSV file and return a tuple of array of data """
        try:
            data = pd.read_csv(file)
            return np.array(data['Prec']), np.array(data['ETP']), np.array(data['Q_obs'])
        except: 
            print('Error : CSV file not loaded...')
    
    def exec(self, file, x):
        """ Execute ModHyPMA algorithm """

        try:
            self.m = x[0]
            self.l = x[1]
            self.tx = x[2]
            self.p2 = x[3]

            # Getting Rainfall and PET data from data
            P, ETP, q_obs = self.loadCsv(file)

            # Computation of effective rainfall
            P = [float(x) for x in P]
            q = [max(0,x) for x in P - ETP]

            # Computation of soil's states
            size = len(q)
            X = [0.0]
            
            for i in range(1,size):
                try:
                    if q[i] == 0:
                        X.append(X[i-1] - (self.m/self.l) * X[i-1])
                    else:
                        X.append(X[i-1] + (self.m/self.l) * q[i]**(2*self.m - self.p2))
                except:
                    print("Error of soil state computation : Bad dataset !")

            # Limitation of the state of soil by introduce TX
            for x in X:
                if x < self.tx:
                    x = 0
            
            # Computation of q_sim
            q_sim = [0]
            
            for i in range(1,size):
                nx = 0
                try:
                    nx = max(q_sim[i-1] - (self.m/ self.l) * q_sim[i-1]**(2*self.m - 1) + X[i]*q[i-1]/self.l, 0)
                except: 
                    print("Error of debits computation : Bad set of parameters  !")
                finally:
                    q_sim.append(nx)

            return q_obs, q_sim
        except:
            print(x)

    def calibrate(self, crit, meth):
        """
            Calibration  de Nelder Mead
            Algorithme
                final_simplex = nelder_mead(initial_simplex, objective_function)
        """
        """ 
            La fontion calibrate retourne un dictionnaire qui a trois clés : 
                . 'res'      : Résultat de l'optimisation
                . 'q'        :Un tuple de débits observés et simulés
                . 'perf_val' : Valeur du critère de performance
        """
        print('\nStarting calibration ...\n')
        fct = lambda x: self.simulate(x, crit)
        res = minimize(fct, [self.m, self.l, self.tx, self.p2], method=meth)
        q_obs, q_sim = self.exec(self.c_file, res.x)
        print('Parameters values : \n\t - mu : {}\n\t - lambda : {}\n\t - TX : {}\n\t - P2: {}'.format(
            res.x[0], res.x[1], res.x[2], res.x[3] 
        ))
        print('Perform criteria value C : ' + str(fct(res.x)))
        print('\n Calibration terminated with succes.')
        return {
            'res': res,
            'q': (np.array(q_obs), np.array(q_sim)),
            'perf_val': fct(res.x)
        }
    
    def validate(self, crit, meth, f):
        print('\nStarting validation ...\n')
        cal = self.calibrate(crit, meth)
        a,b = self.exec(self.v_file, cal['res'].x)
        print('Perform criteria value V: ' + str(f(a, b)))
        return {
            'cal': cal,
            'q': (a, b),
            'perf_val': f(a, b)
        }
    
    def simulate(self, x, fct):
        q = self.exec(self.file, x)
        return fct(q[0],q[1])

    