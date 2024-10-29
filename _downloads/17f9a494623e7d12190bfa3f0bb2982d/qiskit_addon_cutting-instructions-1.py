import numpy as np
from qiskit import QuantumCircuit
from qiskit_addon_cutting.instructions import Move

qc = QuantumCircuit(4)
qc.ryy(np.pi / 4, 0, 1)
qc.rx(np.pi / 4, 3)
qc.append(Move(), [1, 2])
qc.rz(np.pi / 4, 0)
qc.ryy(np.pi / 4, 2, 3)
qc.append(Move(), [2, 1])
qc.ryy(np.pi / 4, 0, 1)
qc.rx(np.pi / 4, 3)
qc.draw("mpl")