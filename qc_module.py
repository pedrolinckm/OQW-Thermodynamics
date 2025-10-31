#quantum_computation_module
from math import comb, ceil, log2, sqrt, pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.arithmetic.adders import CDKMRippleCarryAdder
import numpy as np
import math
from scipy.special import erf

def bits_needed(max_value: int) -> int:
    return max(1, ceil(log2(max_value+1)))

def bits_big_endian(x: int, width: int):
    bit_list = [(x >> i) & 1 for i in range(width)]
    bit_list.reverse()
    return bit_list

def bits_little_endian(x: int, width: int):
    bit_list = [(x >> i) & 1 for i in range(width)]
    return bit_list


def add_controls_equal(circ: QuantumCircuit, a_reg, value_bits, *, big_endian=True):
    n = len(a_reg)

    reg_qubits = list(a_reg) if big_endian else list(a_reg)[::-1]

    flips = []
    for qb, bit in zip(reg_qubits, value_bits):
        if bit == 0:
            circ.x(qb)
            flips.append(qb)

    def unflip():
        for qb in flips:
            circ.x(qb)

    return unflip

def quantum_sum_gate(n):
    cin = QuantumRegister(1,'cin')
    cout = QuantumRegister(1,'cout')
    a = QuantumRegister(n,'a')
    b = QuantumRegister(n,'b')
    qc = QuantumCircuit(cin,a,b,cout)
    for k in range(n//2):
        qc.swap(a[k], a[n - 1 - k])
        qc.swap(b[k], b[n - 1 - k])

    qc = qc.compose(CDKMRippleCarryAdder(num_state_qubits=n))
    for k in range(n//2):
        qc.swap(a[k], a[n - 1 - k])
        qc.swap(b[k], b[n - 1 - k])
    return qc


def quantum_sub_gate(n: int):
    qin   = QuantumRegister(1,  "qin")
    qA    = QuantumRegister(n,  "A")
    qB    = QuantumRegister(n,  "B")
    qout  = QuantumRegister(1,  "qout")
    qc = QuantumCircuit(qin, qA, qB, qout, name=f"QSUB_{n}")

    # ~A e cin' = 1 - bin
    for i in range(n):
        qc.x(qA[i])
    qc.x(qin[0])

    # soma: (qin, A, B, qout) — aceita Gate/Instruction/Circuit
    sum_op = quantum_sum_gate(n)
    try:
        sum_inst = sum_op.to_instruction()  # se for QuantumCircuit
    except AttributeError:
        sum_inst = sum_op                   # se já for Gate/Instruction
    qc.append(sum_inst, [qin[0], *qA, *qB, qout[0]])

    # restaurar entradas externas
    for i in range(n):
        qc.x(qA[i])
    qc.x(qin[0])

    # borrow_out = NOT(carry_out)
    qc.x(qout[0])
    
    return qc.to_instruction()


def steady_state_linear_oqw(omega:float, N: int):
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
    x = np.asarray(x)
    return np.piecewise(
        x,
        [x >= 3/2, x <= -3/2],
        [1, -1, lambda x: 2/3 * x]
    )


def S_G(N,omega,t):
    v = 2*omega - 1
    part_one = np.log(2*pi*t)/(4*np.sqrt(2*pi)) * ( 1 + erf( (N-v*t)/(np.sqrt(2*t)) ) )
    part_two = -(1/(2*np.sqrt(2*pi))) * (N-v*t)/(np.sqrt(t)) * np.exp( -(N-v*t)**2/(2*t) )
    part_three = 1/(2*np.sqrt(2*pi)) * ( erf( (N-v*t)/(np.sqrt(2*t)) ) + np.sqrt(pi/2))
    result = part_one + part_two + part_three
    return result


def S_corrected(N,omega,t):
    v = 2*omega - 1 
    s_g = S_G(N,omega,t)
    s_ss = - 1/2 * (1 - erf( (N-v*t)/np.sqrt(2*t) )) * np.log(1/2 * (1 - erf( (N-v*t)/np.sqrt(2*t) )))
    s_total = s_g + s_ss
    return s_total


def Prob(N,omega,t):
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


def S_entropy(N,omega,t):
    prob_list = Prob(N,omega,t)
    S = 0
    for m in range(N):
        S += - prob_list[m, t] * np.log(prob_list[m, t] + 1e-15)
    return S