import os
import win32com.client
import matplotlib.pyplot as plt


def graphique(taille, titre_sim, titre_fig, titrex, titrey, no_fig, x, y, sheet, cell):
    plt.figure(dpi=taille)
    # TODO verifier que les str n'ont pas d'espaces
    nom_fichier = titre_sim + no_fig
    plt.title(titre_fig+'\n' + f'simulation {titre_sim}')
    plt.plot(x, y)
    plt.xlabel(titrex)
    plt.ylabel(titrey)
    plt.grid('on')
    path_fig = os.path.join(titre_sim, nom_fichier)
    plt.savefig(path_fig)
    fig_cell = sheet.Range(cell)

    fig = sheet.Pictures().Insert(path_fig + '.png')
    fig.Height = 350
    fig.Left = fig_cell.Left
    fig.Top = fig_cell.Top
