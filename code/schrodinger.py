from scipy.integrate import solve_bvp
import numpy as np

# TODO: Définir l'équation différentielle de Schrödinger à résoudre avec solve_bvp

def equation_schrodinger(x, y, V, m):
    """
    y[0] = psi, y[1] = dpsi/dx
    """
    hbar = 1.054571817e-34
    return np.vstack((y[1], 2*m*(V(x) - E)/hbar**2 * y[0]))  # à ajuster
