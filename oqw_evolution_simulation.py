#Simulation of OQW evolution (Fig.4)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#font_path = '/home/pedro-linck/Downloads/times.ttf'  # Update this path if necessary
#font_prop = fm.FontProperties(fname=font_path)

#plt.rcParams['font.family'] = 'serif'

#OQW parameters
omega = 0.9  
lambd = 1 - omega
a = omega / lambd

# Define N (size of graph) and Nmax (number of steps)
N = 300   # Maximum value of m
Nmax = 400  # Maximum value of n

# Initialize P array
P = np.zeros((N, Nmax+1))

# Set initial condition
P[0, 0] = 1

# Compute P[m, n] recursively
for n in range(1, Nmax+1):
    for m in range(N):
        if m == 0:
            P[m, n] = lambd * (P[m, n-1] + P[m+1, n-1])
        elif m == N - 1:
            P[m, n] = omega * (P[m, n-1] + P[m-1, n-1])
        else:
            P[m, n] = omega * P[m-1, n-1] + lambd * P[m+1, n-1]


plt.figure(figsize=(10, 6))
for n in range(100, Nmax+1, 50):
    plt.plot(range(N), P[:, n], marker='o', label=f'n={n}')
plt.xlabel('m')
plt.ylabel('P(m, n)')
plt.legend()
plt.grid(True)

plt.savefig('P_vs_m_for_selected_n.png')    

plt.savefig('P_vs_m_for_selected_n.pgf')