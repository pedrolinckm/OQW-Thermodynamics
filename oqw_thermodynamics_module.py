#quantum_computation_module
from math import comb, ceil, log2, sqrt, pi
import numpy as np
import math
from scipy.special import erf


def steady_state_linear_oqw(omega:float, N: int):
    '''
    Returns the steady state probability distribution for linear OQW with given omega and N
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


def mean_energy(N,omega, epsilon):
    '''
    Returns the mean energy for linear OQW with given omega, N and epsilon
    '''
    beta =  -np.log(omega/(1-omega)) / epsilon
    E = epsilon/(np.exp(beta * epsilon)-1) - N * epsilon / (np.exp(N * beta * epsilon)-1)
    return E

def standart_deviation_energy(N,omega,epsilon):
    beta = -np.log(omega/(1-omega)) / epsilon
    var_E = (epsilon**2 * np.exp(beta * epsilon)) / (np.exp(beta * epsilon)-1)**2 - (N**2 * epsilon**2 * np.exp(N * beta * epsilon)) / (np.exp(N * beta * epsilon)-1)**2
    std_E = np.sqrt(var_E)
    return std_E


def S_G(N,omega,t):
    '''
    Returns the Von Neumann entropy approximation for linear OQW with given omega, N and t
    ''' 
    v = 2*omega - 1
    part_one = np.log(2*pi*t)/(4*np.sqrt(2*pi)) * ( 1 + erf( (N-v*t)/(np.sqrt(2*t)) ) )
    part_two = -(1/(2*np.sqrt(2*pi))) * (N-v*t)/(np.sqrt(t)) * np.exp( -(N-v*t)**2/(2*t) )
    part_three = 1/(2*np.sqrt(2*pi)) * ( erf( (N-v*t)/(np.sqrt(2*t)) ) + np.sqrt(pi/2))
    result = part_one + part_two + part_three
    return result


def S_G_corrected(N,omega,t):
    v = 2*omega - 1
    return 0.5 * ((1 + np.log(2 * np.pi * t)) * 0.5 * (1 + erf((N - v * t) / np.sqrt(2 * t))) - (N - v * t) / np.sqrt(t) / np.sqrt(2 * np.pi) * np.exp(-((N - v * t)**2) / (2 * t)))


def S_corrected(N,omega,t):
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
    Returns the probability distribution P[m, n] for linear OQW with given omega, N and t
    '''
    # Compute P[m, n] recursively
    # Initialize P array
    P = np.zeros((N, t+1))
    # Set initial condition
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


def Prob_approximation(N,omega,x,t):
    '''
    Returns the probability distribution approximation P(x, t) for linear OQW with given omega, N and t'''
    v = 2*omega - 1
    P = 0
    if x < N - 2*standart_deviation_energy(N,omega,1)*t:
        P = 1/np.sqrt(2*pi*t) * np.exp( - (x - v*t)**2 / (2*t) )
    else:
        N1 = N - 2*standart_deviation_energy(N,omega,1)*t
        a = omega / (1 - omega)
        Z = (a**N-1)/(a-1)
        p_ss = a**x / Z
        P = 1/2 * (1 - erf( (N1-v*t)/np.sqrt(2*t) )) * p_ss
    return P


def S_entropy(N,omega,t):
    '''
    Returns the entropy for linear OQW with given omega, N and t
    '''
    prob_list = Prob(N,omega,t)
    S = 0
    for m in range(N):
        S += - prob_list[m, t] * np.log(prob_list[m, t] + 1e-15)
    return S


def t_start(N,omega):
    '''
    Returns the starting time for steady state approximation for linear OQW with given omega and N
    '''
    v = 2*omega - 1
    t_start = ( (np.sqrt(1+v*N)-1)/v )**2
    return t_start

def t_end(N,omega):
    '''
    Returns the ending time for steady state approximation for linear OQW with given omega and N
    '''
    v = 2*omega - 1
    t_end = ( (np.sqrt(1+v*N)+1)/v )**2
    return t_end


def S_a_2(N,omega,t):
        v = 2*omega - 1
        a = omega / (1 - omega)
        p_tot_ss = 1/2 * (1 - erf( (N-v*t)/np.sqrt(2*t) ))
        p_ss = []
        n1 = 0
        n2 = N
        Z1 = (a**(n2-n1) - 1)/(a-1)

        p0 = p_tot_ss / Z1

        for m in range(n2-n1):
            p_ss.append(p0 * a**m)
        S_ss = 0
        for m in range(n2-n1):
            S_ss += - p_ss[m] * np.log(p_ss[m] + 1e-15)
        s_g = S_G_corrected(mean_energy(N,omega,1)-2*standart_deviation_energy(N,omega,1),omega,t)
        s_total = s_g + S_ss
        return s_total
