import os
import matplotlib.pyplot as plt


def graphique(taille, titre_sim, titre_fig, titrex, titrey, no_fig, x, y, sheet, cell):
    plt.figure(dpi=taille)
    nom_fichier = titre_sim + no_fig
    plt.title(titre_fig + '\n' + f'Simulation {titre_sim}')
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


def graphrendturb(taille, titre_sim, titre_fig, titrex, titrey, no_fig, mdot, rend, sheet, cell):
    plt.figure(dpi=taille)
    nom_fichier = titre_sim + no_fig
    plt.title(titre_fig + '\n' + f'Simulation {titre_sim}')
    plt.plot(mdot, rend[0], label="Rendement HP")
    plt.plot(mdot, rend[1], label="Rendement LP")
    plt.xlabel(titrex)
    plt.ylabel(titrey)
    plt.grid('on')
    plt.legend()
    path_fig = os.path.join(titre_sim, nom_fichier)
    plt.savefig(path_fig)
    fig_cell = sheet.Range(cell)

    fig = sheet.Pictures().Insert(path_fig + '.png')
    fig.Height = 350
    fig.Left = fig_cell.Left
    fig.Top = fig_cell.Top
