import matplotlib.pyplot as plt

# TODO: Créer des fonctions de visualisation pour les résultats

def plot_potentiel(x, V):
    plt.plot(x, V)
    plt.title("Potentiel total")
    plt.xlabel("x (m)")
    plt.ylabel("V(x) (J)")
    plt.grid()
    plt.show()
