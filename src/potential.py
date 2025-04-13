import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURATION GÉNÉRALE ===
unit = "eV"         # "eV" ou "J"
save_fig = True    # Si True → sauvegarde, sinon affiche
test_mode = False    # Si True → rapide, sinon haute précision

# === PARAMÈTRES PHYSIQUES ===
epsilon_0 = 8.854187817e-12     # Permittivité du vide (F/m)
e = 1.602176634e-19             # Charge élémentaire (C)
k = 1 / (4 * np.pi * epsilon_0) # Constante de Coulomb (N·m²/C²)
eV_to_J = e                     # Conversion 1 eV → J

# === PARAMÈTRES DU POTENTIEL ===
Q = 1                           # Charge en unité de e
a = 0.01                        # Adoucissement (unités arbitraires)
L = 0.05                        # Période entre les puits
N = 10                          # Nombre de puits
voisinage = 1                   # Nombre de voisins pris en compte

# === GRILLE NUMÉRIQUE ===
if test_mode:
    x_points = 400
else:
    x_points = 4000
x = np.linspace(0, (N + 1) * L, x_points)  # Grille spatiale


# === UTILITAIRES ===
def convert_energy(val):
    """Convertit une énergie en fonction de l’unité choisie."""
    if unit == "eV":
        return val
    elif unit == "J":
        return val * eV_to_J
    else:
        raise ValueError("Unité non supportée : 'eV' ou 'J'")


def coulomb_softened(x, x0, Q, a):
    """Potentiel Coulombien adouci centré en x0, retourné en eV."""
    r = np.sqrt((x - x0)**2 + a**2)
    V = - Q * e**2 * k / r      # Potentiel en J
    return V / e                # Conversion en eV (plus stable)


def V_total(x, Q=Q, a=a, L=L, N=N, voisinage=voisinage):
    """Calcule le potentiel total avec voisins autour du puits le plus proche."""
    V = 0
    j_c = int(np.floor(x / L - 0.5))
    j_c = max(0, min(N - 1, j_c))  # Clamp dans [0, N-1]

    for j in range(j_c - voisinage, j_c + voisinage + 1):
        if 0 <= j < N:
            xj = (j + 1) * L
            V += coulomb_softened(x, xj, Q, a)
    return V


# === CALCUL DU POTENTIEL SUR LA GRILLE ===
V = np.array([V_total(xi) for xi in x])      # Potentiel en eV
V_converted = convert_energy(V)              # Converti en eV ou J

# === AFFICHAGE CONSOLE ===
print(f"V_min = {np.min(V_converted):.3e} {unit}, V_max = {np.max(V_converted):.3e} {unit}")

# === PLOT ===
plt.figure()
plt.plot(x, V_converted)
plt.xlabel("Position [a.u.]")
plt.ylabel(f"Énergie potentielle [{unit}]")
plt.title("Profil de potentiel adouci")
plt.grid(True)

# === SAUVEGARDE OU AFFICHAGE ===
if save_fig:
    output_dir = os.path.join("C:", os.sep, "Users", "balia", "OneDrive - Universite de Liege",
                              "Bureau", "universite", "bac2", "Q2", "phys4", 
                              "project", "phys2026-projet-schrodinger", "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "potential.pdf")
    plt.savefig(output_path)
    print(f"Figure sauvegardée dans {output_path}")
else:
    plt.show()
