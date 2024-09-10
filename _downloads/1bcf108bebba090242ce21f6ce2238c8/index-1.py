import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import RZZGate
from qiskit_addon_cutting.qpd import QPDBasis

theta_values = np.linspace(0, np.pi, 101)
bases = [QPDBasis.from_instruction(RZZGate(theta)) for theta in theta_values]

colors = ["#57ffff", "#2B568C", "#007da3", "#ffa502", "#7abaff", "#f2cc86"]
labels = ['$I \otimes I$ ','$Z \otimes Z$','$M_z \otimes S$','$-M_z \otimes S^\dagger$','$S \otimes M_z$','$-S^\dagger \otimes M_z$']
plt.stackplot(theta_values, *zip(*[np.abs(basis.coeffs) for basis in bases]), labels=labels, colors=colors)
plt.axvline(np.pi / 2, c="#aaaaaa", linestyle="dashed")
plt.axvline(np.pi / 4, c="#aaaaaa", linestyle="dotted")
plt.axhline(1, c="#aaaaaa", linestyle="solid")
plt.legend(loc='upper right')
plt.xlim(0, np.pi)
plt.ylim(0, 3.6)
plt.xlabel(r"RZZGate rotation angle $\theta$")
plt.ylabel("Absolute coefficients, stacked (sum = $\gamma$)")
plt.title("Quasiprobability decomposition for RZZGate")
plt.gca().set_xticks(np.linspace(0, np.pi, 5))
plt.gca().set_xticklabels(['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
plt.annotate("CXGate\nfamily", (np.pi / 2, 3), textcoords="offset points", xytext=(-5, 10), ha="right")
plt.annotate("CSGate\nfamily", (np.pi / 4, 1 + np.sqrt(2)), textcoords="offset points", xytext=(-5, 10), ha="right")