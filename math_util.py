import numpy as np
import matplotlib.pyplot as plt


def biv_poly(x, y, c):
    somme = 0
    for j in range(c.shape[0]):
        for i in range(c.shape[1]):
            somme += c[j, i] * x ** i * y ** j
    return somme


def racine(x, y, xg, xd):
    fxg = np.interp(xg, x, y)
    fxd = np.interp(xd, x, y)
    tol = 1

    if fxg*fxd >= 0:
        r = "Optimisation: Pas de zéro trouvé dans l'interval donné"
    else:
        r = (xg+xd)/2
        while abs(xg-xd)/2 > tol:
            r = (xg+xd)/2
            fr = np.interp(r, x, y)
            if fxg*fr < 0:
                xd = r
            else:
                xg = r
    return r


def point_optimal(x, y, low, high):
    der = np.gradient(y, x, edge_order=2)
    point = racine(x, der, low, high)
    return point


def graphique(x, y, titre, titrex, titrey):
    plt.figure(dpi=300)
    plt.plot(x, y)
    plt.xlabel(titrex)
    plt.ylabel(titrey)
    plt.title(titre)
    plt.show()


def frac_mdot_sout(fractions, index):

    if index not in range(0, len(fractions)):
        return None
    else:
        produit = fractions[index]
        for i in range(0, index):
            # print(f'{produit} x (1 - {fractions[i]})')
            produit = produit*(1-fractions[i])
        return produit


def frac_mdot_turbine(fractions, index):
    """attention a l'index """
    produit = 1
    if index not in range(0, len(fractions)):
        return None
    else:
        for i in range(0, index + 1):
            # print(f'{produit} x (1 - {fractions[i]})')
            produit = produit*(1-fractions[i])

        return produit
