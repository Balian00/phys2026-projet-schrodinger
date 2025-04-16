import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION GÉNÉRALE ===
unit = "eV"         # Unité d'énergie : "eV" ou "J"
test_mode = True    # Mode test (rapide) ou haute précision

# === CONSTANTES PHYSIQUES ===
epsilon_0 = 8.854187817e-12     # Permittivité du vide (F/m)
e = 1.602176634e-19             # Charge élémentaire (C)
k = 1 / (4 * np.pi * epsilon_0) # Constante de Coulomb (N·m²/C²)
eV_to_J = e                     # Conversion 1 eV → J

# === PARAMÈTRES DU POTENTIEL ===
Q = 1                           # Charge source
a = 0.01                        # Paramètre d'adoucissement
L = 0.05                        # Distance entre charges
N = 10                          # Nombre de charges
voisinage = N                   # Nombre de voisins considérés

# === GRILLE NUMÉRIQUE ===
x_points = 400 if test_mode else 4000
n_marches = 100 if test_mode else 500
x = np.linspace(0, (N + 1) * L, x_points)  # Grille de positions

# === CONVERSIONS ===
def convert_energy(val):
    """Convertit l'énergie en fonction de l'unité choisie."""
    if unit == "eV":
        return val
    elif unit == "J":
        return val * eV_to_J
    else:
        raise ValueError("Unité non supportée : 'eV' ou 'J'")

# === POTENTIEL COULOMBIEN ADOUCI ===
def coulomb_softened(x, x0, Q, a):
    """Calcule le potentiel de Coulomb adouci."""
    r = np.sqrt((x - x0)**2 + a**2)
    V = - Q * e**2 * k / r
    return V / e  # Retourne en eV

def V_total(x, Q=Q, a=a, L=L, N=N, voisinage=voisinage):
    """Calcule le potentiel total en sommant les contributions des charges voisines."""
    V = 0
    j_c = int(np.floor(x / L - 0.5))  # Index de la charge centrale
    j_c = max(0, min(N - 1, j_c))    # Limite l'index aux bornes
    for j in range(j_c - voisinage, j_c + voisinage + 1):
        if 0 <= j < N:
            xj = (j + 1) * L
            V += coulomb_softened(x, xj, Q, a)
    return V

# === CALCUL DU POTENTIEL CONTINU SUR LA GRILLE ===
V = np.array([V_total(xi) for xi in x])
V[0] = 0.0  # Condition aux bords
V[-1] = 0.0
V_converted = convert_energy(V)

print(f"V_min = {np.min(V_converted):.3e} {unit}, V_max = {np.max(V_converted):.3e} {unit}")

# === DISCRÉTISATION DU POTENTIEL ===
def discretize_potential_with_boundaries(x, V_continuous, n):
    """
    Discrétise une fonction en n marches + 2 bords à 0.
    Retourne les centres des marches et les valeurs discrètes.
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
            V_discrete.append(0.0)  # Conditions aux bords
        else:
            V_avg = np.mean(V_continuous[mask])
            V_discrete.append(V_avg)

    return x_centers, np.array(V_discrete)

# === CONSTRUCTION DE LA FONCTION DISCRÈTE ===
def get_edges_from_centers(x_centers):
    """Reconstruit les bords à partir des centres des intervalles."""
    dx = x_centers[1] - x_centers[0]
    x_edges = np.concatenate(([x_centers[0] - dx / 2], x_centers + dx / 2))
    return x_edges

def V_discret_function(x_vals, x_edges, V_vals):
    """
    Évalue la fonction discrète par morceaux sur x_vals.
    Associe chaque x à la marche correspondante.
    """
    indices = np.searchsorted(x_edges, x_vals, side='right') - 1
    indices = np.clip(indices, 0, len(V_vals) - 1)  # Évite les dépassements
    return V_vals[indices]

# === APPEL DE LA DISCRÉTISATION ===
x_marches, V_marches = discretize_potential_with_boundaries(x, V_converted, n_marches)

# === CONSTRUCTION DE LA VERSION FONCTIONNELLE PAR MORCEAUX ===
x_edges = get_edges_from_centers(x_marches)
x_fine = np.linspace(x[0], x[-1], 5000)
V_piecewise = V_discret_function(x_fine, x_edges, V_marches)

# === AFFICHAGE DU POTENTIEL CONTINU VS DISCRET ===
plt.figure()
plt.plot(x, V_converted, label="Potentiel continu")
plt.plot(x_fine, V_piecewise, color='orange', label="Potentiel discrétisé")
plt.xlabel("Position [a.u.]")
plt.ylabel(f"Énergie potentielle [{unit}]")
plt.title("Potentiel continu vs discrétisé")
plt.grid(True)
plt.legend()
if test_mode:
    plt.show()
else:
    plt.savefig("pot_discretisation.pdf")

# === COMPARAISON AVEC DIFFÉRENTS VOISINAGES ===
plt.figure(figsize=(8, 6))

# Potentiel avec N voisins
plt.plot(x, V_converted, label="Potentiel continu (N voisins)", color='blue')

# Potentiel avec 1 voisin
voisinage = 1
V_1_voisin = np.array([V_total(xi, Q=Q, a=a, L=L, N=N, voisinage=voisinage) for xi in x])
V_1_voisin_converted = convert_energy(V_1_voisin)
plt.plot(x, V_1_voisin_converted, label="Potentiel continu (1 voisin)", color='red')

# Configuration du graphe
plt.xlabel("Position [a.u.]")
plt.ylabel(f"Énergie potentielle [{unit}]")
plt.title("Comparaison des potentiels : N voisins vs 1 voisin")
plt.grid(True)
plt.legend()

# Affichage ou sauvegarde
if test_mode:
    plt.show()
else:
    plt.savefig("pot_comparaison_N_et_1_voisin_sur_meme_graphe.pdf")
