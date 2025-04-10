import numpy as np
from code.potentiel import potentiel_coulombien_adouci
from code.analyse import plot_potentiel

# Exécution d'un test de potentiel
x = np.linspace(-1e-9, 1e-9, 1000)
V = potentiel_coulombien_adouci(x, 0, Q=1, a=0.05e-9)
plot_potentiel(x, V)
