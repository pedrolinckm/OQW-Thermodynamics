#oqw_module.py
from math import comb, ceil, log2, sqrt, pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.arithmetic.adders import CDKMRippleCarryAdder
import numpy as np
import math
from scipy.special import erf


def steady_state_linear_oqw(omega:float, N: int):
    '''
    Function that computes the steady state probabilities for a linear open quantum walk with N nodes and coin parameter omega.
    Returns a list of probabilities for each node.
    '''
    if omega != 0.5:
        prob_list = []
        a = omega / (1 - omega)
        Z = (a**N-1)/(a-1)
        for m in range(N):
            xm = a**m / Z
            prob_list.append(xm)
    else: 
        return [1/N]*N    
    return prob_list


def erf_approximation(x):
    '''
    Approximation of the error function using piecewise linear functions.
    1 for x >= 3/2
    -1 for x <= -3/2
    (2/3)x for -3/2 < x < 3/2
    '''
    x = np.asarray(x)
    return np.piecewise(
        x,
        [x >= 3/2, x <= -3/2],
        [1, -1, lambda x: 2/3 * x]
    )


def mean_energy(N,omega, epsilon):
    '''
    Mean energy of the system.
    '''
    beta =  -np.log(omega/(1-omega)) / epsilon
    E = epsilon/(np.exp(beta * epsilon)-1) - N * epsilon / (np.exp(N * beta * epsilon)-1)
    return E

def standart_deviation_energy(N,omega,epsilon):
    '''
    Standard deviation of the energy of the system.
    '''
    beta = -np.log(omega/(1-omega)) / epsilon
    var_E = (epsilon**2 * np.exp(beta * epsilon)) / (np.exp(beta * epsilon)-1)**2 - (N**2 * epsilon**2 * np.exp(N * beta * epsilon)) / (np.exp(N * beta * epsilon)-1)**2
    std_E = np.sqrt(var_E)
    return std_E


def S_G(N,omega,t):
    '''
    Entropy of the Gaussian part of the distribution.
    '''
    v = 2*omega - 1
    part_one = np.log(2*pi*t)/(4*np.sqrt(2*pi)) * ( 1 + erf( (N-v*t)/(np.sqrt(2*t)) ) )
    part_two = -(1/(2*np.sqrt(2*pi))) * (N-v*t)/(np.sqrt(t)) * np.exp( -(N-v*t)**2/(2*t) )
    part_three = 1/(2*np.sqrt(2*pi)) * ( erf( (N-v*t)/(np.sqrt(2*t)) ) + np.sqrt(pi/2))
    result = part_one + part_two + part_three
    return result


def S_G_corrected(N,omega,t):
    '''
    Corrected entropy of the Gaussian part of the distribution.
    '''
    v = 2*omega - 1
    return 0.5 * ((1 + np.log(2 * np.pi * t)) * 0.5 * (1 + erf((N - v * t) / np.sqrt(2 * t))) - (N - v * t) / np.sqrt(t) / np.sqrt(2 * np.pi) * np.exp(-((N - v * t)**2) / (2 * t)))


def S_corrected(N,omega,t):
    '''
    Corrected total entropy of the system.
    '''
    v = 2*omega - 1
    a = omega / (1 - omega)
    Z = (a**N-1)/(a-1)

    p_tot_ss = 1/2 * (1 - erf( (N-v*t)/np.sqrt(2*t) ))

    p_ss = []
    p0 = p_tot_ss / Z

    for m in range(N):
        p_ss.append(p0 * a**m)

    s_ss = 0

    for m in range(N):
        s_ss += - p_ss[m] * np.log(p_ss[m] + 1e-15)
    

    s_g = S_G_corrected(N,omega,t)

    s_total = s_g + s_ss
    return s_total


def Prob(N,omega,t):
    '''
    Function that computes the probability distribution P[m,n] of an open quantum walk on a line with N nodes,
    coin parameter omega, after t time steps.'''

    P = np.zeros((N, t+1))
    P[0, 0] = 1

    lambd = 1 - omega

    for n in range(1, t+1):
        for m in range(N):
            if m == 0:
                P[m, n] = lambd * (P[m, n-1] + P[m+1, n-1])
            elif m == N - 1:
                P[m, n] = omega * (P[m, n-1] + P[m-1, n-1])
            else:
                P[m, n] = omega * P[m-1, n-1] + lambd * P[m+1, n-1]
    return P


def Prob_approximation(N, omega, x, t):
    """
    Approximate probability distribution P(x,t) of an open quantum walk on a line
    with N nodes, coin parameter omega, at position x after t time steps.
    """
    v = 2*omega - 1
    a = omega / (1 - omega)

    # Standard deviation and cutoff
    std = standart_deviation_energy(N, omega, 1)
    N_cut = N - 2*std  # use Gaussian for x < N - 2*std

    # Geometric tail normalization (as in your code)
    n1 = 0
    n2 = N
    Z1 = (a**(n2 - n1) - 1) / (a - 1)  # sum_{k=n1}^{n2-1} a^k

    p_tot_ss = 0.5 * (1 - erf(((mean_energy(N, omega, 1) - 2*std) - v*t) / np.sqrt(2*t)))
    p0 = p_tot_ss / Z1  # base factor so that sum tail = p_tot_ss

    # Piecewise definition: Gaussian for x < N_cut, geometric tail for x >= N_cut
    if x < N_cut:
        p_g = 1 / np.sqrt(2 * pi * t) * np.exp(-(x - v*t)**2 / (2*t))
        p_ss = 0.0
    else:
        p_g = 0.0
        # geometric part ~ a^x (with normalization via p0)
        p_ss = p0 * a**(x - n1)

    P = p_g + p_ss
    return P

def S_entropy(N,omega,t):
    '''
    Function that computes the entropy S(t) of an open quantum walk on a line with N nodes,
    coin parameter omega, after t time steps.
    '''
    prob_list = Prob(N,omega,t)
    S = 0
    for m in range(N):
        S += - prob_list[m, t] * np.log(prob_list[m, t] + 1e-15)
    return S


def t_start(N,omega):
    '''
    Function that computes the starting time t_start for the approximation of the entropy.
    '''
    v = 2*omega - 1
    t_start = ( (np.sqrt(1+v*N)-1)/v )**2
    return t_start

def t_end(N,omega):
    '''
    Function that computes the ending time t_end for the approximation of the entropy.
    '''
    v = 2*omega - 1
    t_start = ( (np.sqrt(1+v*N)+1)/v )**2
    return t_start


def S_a_2(N, omega, t):
    """
    Approximate entropy S(t) using a Gaussian for m < N_cut and steady-state distribution for m >= N_cut.
    """
    v = 2*omega - 1
    a = omega / (1 - omega)
    std = standart_deviation_energy(N, omega, 1)

    N_cut_float = N - 3 * std
    N_cut = int(np.floor(N_cut_float))
    N_cut = max(0, min(N, N_cut))

    p_tot_ss = 0.5 * (1 - erf((N_cut - v*t) / np.sqrt(2*t)))

    if abs(a - 1.0) < 1e-14:
        Z_tail = N - N_cut
    else:
        Z_tail = a**N_cut * (a**(N - N_cut) - 1) / (a - 1)

    p0 = p_tot_ss / Z_tail

    S_ss = 0.0
    for m in range(N_cut, N):
        p = p0 * a**m      
        S_ss += -p * np.log(p + 1e-15)

    S_g = 0.0
    for m in range(0, N_cut):
        p = 1 / np.sqrt(2 * pi * t) * np.exp(-(m - v*t)**2 / (2*t))
        S_g += -p * np.log(p + 1e-15)

    S_total = S_g + S_ss
    return S_total

def S_a_3(N,omega,t):
        '''
        Approximate entropy S(t) using the steady state distribution for m >= N - 2*std_dev_energy.
        '''
        v = 2*omega - 1
        a = omega / (1 - omega)
        p_tot_ss = 1/2 * (1 - erf( (N-v*t)/np.sqrt(2*t) ))
        p_ss = []
        Z1 = (a**(N) - 1)/np.log(a)

        p0 = p_tot_ss / Z1

        for m in range(N):
            p_ss.append(p0 * a**m)
        S_ss = 0
        for m in range(N):
            S_ss += - p_ss[m] * np.log(p_ss[m] + 1e-15)
        s_g = S_G_corrected(mean_energy(N,omega,1)-2*standart_deviation_energy(N,omega,1),omega,t)
        s_total = s_g + S_ss
        return s_total
