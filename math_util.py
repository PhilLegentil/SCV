import numpy as np
import matplotlib.pyplot as plt


def biv_poly(x, y, c):

    somme = 0
    for j in range(c.shape[0]):
        for i in range(c.shape[1]):
            somme += c[j, i] * x ** i * y ** j
    return somme


def racine(coef, xg, xd):
    fxg = np.polyval(coef, xg)
    fxd = np.polyval(coef, xd)

    if fxg*fxd >= 0:
        print('pas de zero dans cet interval')
        r = []
    else:
        r = (xg+xd)/2
        while abs(xg-xd)/2 > 1e-5:
            r = (xg+xd)/2
            fr = np.polyval(coef, r)
            if fxg*fr < 0:
                xd = r
            else:
                xg = r
    return r


def point_optimal(x, y, low, high):
    der = np.gradient(y, x, edge_order=2)
    coef_der = np.polyfit(x, der, 4)
    return racine(coef_der, low, high)


def fonction_poly(x, y, x_0):
    coef = np.polyfit(x, y, 4)
    return np.polyval(coef, x_0)


def graphique(x, y, titre, titrex, titrey):
    plt.figure(dpi=300)
    plt.plot(x, y)
    plt.xlabel(titrex)
    plt.ylabel(titrey)
    plt.title(titre)
    plt.show()
