#quantum_computation_module
from math import comb, ceil, log2   
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.arithmetic.adders import CDKMRippleCarryAdder

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