import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURATION GÉNÉRALE ===
unit = "eV"         # "eV" ou "J"
save_fig = False    # Si True → sauvegarde, sinon affiche
test_mode = True    # Si True → rapide, sinon haute précision

# === CONSTANTES PHYSIQUES ===
epsilon_0 = 8.854187817e-12     # Permittivité du vide (F/m)
e = 1.602176634e-19             # Charge élémentaire (C)
k = 1 / (4 * np.pi * epsilon_0) # Constante de Coulomb (N·m²/C²)
eV_to_J = e                     # Conversion 1 eV → J

# === PARAMÈTRES DU POTENTIEL ===
Q = 1
a = 0.01
L = 0.05
N = 10
voisinage = 1

# === GRILLE NUMÉRIQUE ===
x_points = 400 if test_mode else 4000
n_marches = 100 if test_mode else 500
x = np.linspace(0, (N + 1) * L, x_points)

# === CONVERSIONS ===
def convert_energy(val):
    if unit == "eV":
        return val
    elif unit == "J":
        return val * eV_to_J
    else:
        raise ValueError("Unité non supportée : 'eV' ou 'J'")

# === POTENTIEL COULOMBIEN ADOUCI ===
def coulomb_softened(x, x0, Q, a):
    r = np.sqrt((x - x0)**2 + a**2)
    V = - Q * e**2 * k / r
    return V / e  # en eV

def V_total(x, Q=Q, a=a, L=L, N=N, voisinage=voisinage):
    V = 0
    j_c = int(np.floor(x / L - 0.5))
    j_c = max(0, min(N - 1, j_c))
    for j in range(j_c - voisinage, j_c + voisinage + 1):
        if 0 <= j < N:
            xj = (j + 1) * L
            V += coulomb_softened(x, xj, Q, a)
    return V

# === CALCUL DU POTENTIEL CONTINU SUR LA GRILLE ===
V = np.array([V_total(xi) for xi in x])
V[0] = 0.0
V[-1] = 0.0
V_converted = convert_energy(V)

print(f"V_min = {np.min(V_converted):.3e} {unit}, V_max = {np.max(V_converted):.3e} {unit}")

# === DISCRÉTISATION DU POTENTIEL (AVEC CONDITIONS AUX LIMITES) ===
def discretize_potential_with_boundaries(x, V_continuous, n):
    """
    Discrétise une fonction en n marches + 2 bords à 0.
    Retourne:
        - x_centers: centres des marches (n+2 valeurs)
        - V_discrete: hauteurs constantes par marche (n+2 valeurs)
    """
    n_total = n + 2
    x_edges = np.linspace(x[0], x[-1], n_total + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2

    V_discrete = []

    for i in range(n_total):
        x_start = x_edges[i]
        x_end = x_edges[i + 1]
        mask = (x >= x_start) & (x < x_end)
        if i == 0 or i == n_total - 1:
            V_discrete.append(0.0)  # conditions aux bords
        else:
            V_avg = np.mean(V_continuous[mask])
            V_discrete.append(V_avg)

    return x_centers, np.array(V_discrete)


# === APPEL DU TEST DE DISCRÉTISATION ===
x_marches, V_marches = discretize_potential_with_boundaries(x, V_converted, n_marches)

# === AFFICHAGE GRAPHIQUE ===
plt.figure()
plt.plot(x, V_converted, label='Potentiel continu')
plt.step(x_marches, V_marches, where='mid', color='orange', label=f'{n_marches} marches')
plt.xlabel("Position [a.u.]")
plt.ylabel(f"Énergie potentielle [{unit}]")
plt.title("Potentiel adouci vs discrétisé avec V(0)=V(L)=0")
plt.grid(True)
plt.legend()

# === SAUVEGARDE OU AFFICHAGE ===
if save_fig:
    output_dir = os.path.join("C:", os.sep, "Users", "balia", "OneDrive - Universite de Liege",
                              "Bureau", "universite", "bac2", "Q2", "phys4", 
                              "project", "phys2026-projet-schrodinger", "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "potential_discretise.pdf")
    plt.savefig(output_path)
    print(f"Figure sauvegardée dans {output_path}")
else:
    plt.show()
