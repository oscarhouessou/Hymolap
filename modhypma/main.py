from modhypmahydro import modhypma, criteria, algorithm, cplot

# Exemple d'utilisation de la classe ModHyPMA

# Créer une instance de la classe. Le constructeur de la classe
# prend 3 paramètres : le simple initial, le fichier de calage et le fichier de validation
mod = modhypma.ModHyPMA(
   [1.12,35.58,0.15,0.42], 
    'csv/beterou_62_72_cal.csv', 
    'csv/beterou_73_79_val.csv'
    )

# Calage
"""
    Les 10 critères de performances sont implémentés dans le module criteria
    8 sont utilisables pour le calage
    
    La fontion calibrate retourne un dictionnaire qui a trois clés : 
        . 'res'      : Résultat de l'optimisation
        . 'q'        :Un tuple de débits observés et simulés
        . 'perf_val' : Valeur du critère de performance
"""
resultat_calage = mod.calibrate(criteria.combine_mae, algorithm.nelder_mead)


res_opt = resultat_calage['res']
debits = resultat_calage['q']
valeurCritere = resultat_calage['perf_val']
#valeurCritere = 1 - resultat_calage['perf_val'] #Avec le critère nse et r2 il faut prendre 1- resultat_calage['perf_val']
#valeurCritere = resultat_calage['perf_val']
bilC = criteria.bilan(debits[0],debits[1]) # Détermination de la valeur de critère de bilan en calage

# print(res_opt)
# print(debits)
print('Valeur critere en Calage: ' + str(valeurCritere))
print('valeur de critère de Bilan en calage:' +str(bilC)) #Affichage de la valeur du bilan en calage

# Exemple de tracé des courbes à l'issu d'un calage
# Les fonctions de tracé de courbes sont dans le module cplot

print(debits)

cplot.plot(debits)