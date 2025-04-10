import numpy as np

def potentiel_coulombien_adouci(x, x0, Q, a):
    """
    Calcule le potentiel Coulombien adouci centré en x0.
    V(x) = -Qe / (4 * pi * eps0 * sqrt((x - x0)^2 + a^2))
    """
    eps0 = 8.854187817e-12  # permittivité du vide
    e = 1.602176634e-19     # charge élémentaire
    return -Q * e / (4 * np.pi * eps0 * np.sqrt((x - x0)**2 + a**2))
