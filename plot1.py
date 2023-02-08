# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:50:17 2022

@author: Kioen
"""
'''
Plot curves for \mu = 0
'''
import pickle
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])
'''
with open("AGDLDS2gamma=0.5", "rb") as fp: # no need
    AGD1 = pickle.load(fp)
'''
n = 5#1, 2, 3, 4, 5

with open("ASGDLDS"+str(n)+"gamma=0.5+", "rb") as fp: 
    ASGD1 = pickle.load(fp)

with open("ASGDLDS"+str(n)+"gamma=0.8+", "rb") as fp: #gamma=0.5_2 will be better
    ASGD2 = pickle.load(fp)
add = ASGD1[1][0]
with open("adapAGDLDS"+str(n)+"gamma=0.5", "rb") as fp:
    AGD1 = pickle.load(fp)
AGD1.insert(0, add)    
with open("adapAGDLDS"+str(n)+"gamma=0.8", "rb") as fp:
    AGD2 = pickle.load(fp)
AGD2.insert(0, add)

with open("ASVRGLDS"+str(n)+"gamma=0.5+", "rb") as fp:
    ASVRG1 = pickle.load(fp)

ASVRG1[0].insert(0, add)
ASVRG1[1].insert(0, add)
ASVRG1[2].insert(0, add)

with open("ASVRGLDS"+str(n)+"gamma=0.8+", "rb") as fp:
    ASVRG2 = pickle.load(fp)

ASVRG2[0].insert(0, add)
ASVRG2[1].insert(0, add)
ASVRG2[2].insert(0, add)  

with open("adapGDLDS"+str(n)+"gamma=0.5", "rb") as fp:
    GD = pickle.load(fp)
GD.insert(0, add)
with open("SGDLDS"+str(n)+"gamma=0.5+", 'rb') as fp:
    SGD = pickle.load(fp)
SGD[1].insert(0, add)
SGD[0].insert(0, add)
SGD[2].insert(0, add)


'''
The first kind of x-axis: the number of iterations
'''
x1 = list(range(len(AGD1)))
x2 = list(range(len(AGD2)))
x3 = list(range(len(ASGD1[1])))
x4 = list(range(len(ASGD2[1])))
x5 = list(range(len(ASVRG1[1])))
x6 = list(range(len(ASVRG2[1])))
x7 = list(range(len(GD)))
x8 = list(range(len(SGD[1])))
'''
The second kind of x-axis: the number of function and gradient evaluations
'''
'''
x1 = np.linspace(0, 4550000, len(AGD1))
x2 = np.linspace(0, 5065000, len(AGD2))
x3 = np.linspace(0, 1472, len(ASGD1[1]))
x4 = np.linspace(0, 1779, len(ASGD2[1]))
x5 = np.linspace(0, 2758572, len(ASVRG1[1])) #5774613, 2877168
x6 = np.linspace(0, 1903097, len(ASVRG2[1]))
x7 = np.linspace(0, 10990000, len(GD))
x8 = np.linspace(0, 3000, len(SGD[1]))
#x9 = np.linspace(0, 10441032, len(SVRG))
'''
'''
x1 = np.linspace(0, 37015000, len(AGD1))
x2 = np.linspace(0, 27685000, len(AGD2))
x3 = np.linspace(0, 4448, len(ASGD1[1]))
x4 = np.linspace(0, 6169, len(ASGD2[1]))
x5 = np.linspace(0, 9733590, len(ASVRG1[1])) #5774613, 2877168
x6 = np.linspace(0, 7045495, len(ASVRG2[1]))
x7 = np.linspace(0, 21945000, len(GD))
x8 = np.linspace(0, 2000, len(SGD[1]))
#x9 = np.linspace(0, 7812000, len(SVRG))
'''
'''
x1 = np.linspace(0, 5390000, len(AGD1))
x2 = np.linspace(0, 5500000, len(AGD2))
x3 = np.linspace(0, 632, len(ASGD1[1]))
x4 = np.linspace(0, 723, len(ASGD2[1]))
x5 = np.linspace(0, 1386943, len(ASVRG1[1])) #5774613, 2877168
x6 = np.linspace(0, 912611, len(ASVRG2[1]))
x7 = np.linspace(0, 10970000, len(GD))
x8 = np.linspace(0, 1000, len(SGD[1]))
#x9 = np.linspace(0, 520800, len(SVRG))
'''
'''
x1 = np.linspace(0, 98775000, len(AGD1))
x2 = np.linspace(0, 98865000, len(AGD2))
x3 = np.linspace(0, 6086, len(ASGD1[1]))
x4 = np.linspace(0, 9069, len(ASGD2[1]))
x5 = np.linspace(0, 2585942, len(ASVRG1[1]))
x6 = np.linspace(0, 9889033, len(ASVRG2[1]))
x7 = np.linspace(0, 21945000, len(GD))
x8 = np.linspace(0, 2000, len(SGD[1]))
'''
'''
x1 = np.linspace(0, 7545000, len(AGD1))
x2 = np.linspace(0, 6295000, len(AGD2))
x3 = np.linspace(0, 2179, len(ASGD1[1]))
x4 = np.linspace(0, 2653, len(ASGD2[1]))
x5 = np.linspace(0, 1220277, len(ASVRG1[1]))
x6 = np.linspace(0, 1043338, len(ASVRG2[1]))
x7 = np.linspace(0, 11005000, len(GD))
x8 = np.linspace(0, 1000, len(SGD[1]))
'''
plt.figure(figsize=(5, 3), dpi=300)


plt.loglog(x1, AGD1, color='midnightblue', label="QAGDadap"r'$(\gamma=0.5)$')
#plt.fill_between(x1, torch.tensor(AGD1)-error, torch.tensor(AGD1)+error)
plt.loglog(x2, AGD2, color='green', label="QAGDadap"r'$(\gamma=0.8)$')
plt.loglog(x3, ASGD1[1], color='orange', label="QASGD"r'$(\gamma=0.5)$')

plt.fill_between(x3, ASGD1[2], ASGD1[0], alpha=0.2, color='orange', linewidth=0)

plt.loglog(x4, ASGD2[1], color='red', label="QASGD"r'$(\gamma=0.8)$')

plt.fill_between(x4, ASGD2[2], ASGD2[0], alpha=0.2, color='red', linewidth=0)

plt.loglog(x5, ASVRG1[1], color='purple', label="QASVRG"r'$(\gamma=0.5)$')

plt.fill_between(x5, ASVRG1[2], ASVRG1[0], alpha=0.2, color='purple', linewidth=0)

plt.loglog(x6, ASVRG2[1], color='gray', label="QASVRG"r'$(\gamma=0.8)$')

plt.fill_between(x6, ASVRG2[2], ASVRG2[0], alpha=0.2, color='gray', linewidth=0)

plt.loglog(x7, GD, color = 'brown', linestyle="-.", label="GDadap")
plt.loglog(x8, SGD[1], color = 'blue', linestyle="-.", label="SGD")

plt.fill_between(x8, SGD[2], SGD[0], alpha=0.2, color='blue', linewidth=0)

#plt.loglog(x9, SVRG, linestyle="--", label="SVRG")

'''
# We do not plot GD, SGD and SVRG when we use the second kind of x-axis
'''
'''
plt.semilogx(x1, AGD, label="QAGD")
plt.semilogx(x2, ASGD, label="QASGD")
plt.semilogx(x3, ASVRG, label="QASVRG")
plt.semilogx(x4, adapAGD, linestyle="-.", label="GD")
plt.semilogx(x5, GD, linestyle="-.", label="SGD")
plt.semilogx(x6, SGD, linestyle="--", label="SVRG")
plt.semilogx(x7, SVRG, linestyle="--", label="SVRG")
'''
plt.title("LDS"+str(n)+" without random perturbation") ###
#plt.xlabel(r'$\log(n)\ $'r'($n$'": Function and Gradient Evaluations)")
plt.xlabel('#(func+grad)')
plt.xlabel("Number of Iterations") ###
#plt.ylabel(r'$\log(f-f^*)$')
plt.ylabel('Loss')
plt.legend(loc='best', fontsize="xx-small")
plt.grid()
plt.savefig("LDS"+str(n)+"iter.png",dpi=300)
plt.show()