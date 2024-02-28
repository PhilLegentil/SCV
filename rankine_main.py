import math_util

print("Démarrage de la simulation...")

import os
from rankine_fct import *
import win32com.client
import matplotlib.pyplot as plt
import numpy as np
import math_util as mut
import pandas as pd
import scipy as sp

print('Lecture des donnees')
file_path = "interface.xlsm"
data = pd.read_excel(file_path, header=0)

inf = data['debit'][0]
sup = data['debit'][1]
n_pts = int(data['debit'][2])

m_dot = np.linspace(inf, sup, n_pts)

# TODO verifier que soutirage != 1

W_net, rendement, eau_alim, vapeur_cond, index_titre_cond, vapeur_chaud, int_nonvap = travail(m_dot, data)
print('Traitement')



titre_sim = f'{data["noms"][0]}'

try:
    os.mkdir(titre_sim)
except:
    pass

ExcelApp = win32com.client.GetActiveObject("Excel.Application")
ExcelApp.Visible = True

workbook = ExcelApp.Workbooks.Open(r"interface.xlsm")
sheet = workbook.Worksheets(2)

imdot_aj_inf = 0
imdot_aj_sup = len(m_dot) - 1
print(vapeur_chaud)
# TODO verifier la logique
if vapeur_cond and vapeur_chaud:

    # TODO highlight
    if index_titre_cond < int_nonvap[-1]:
        imdot_aj_sup = index_titre_cond
        message = f"L'eau en sortie du condenseur est entierement sous forme de vapeur pour mdot >= {m_dot[index_titre_cond]} "
    else:
        imdot_aj_sup = int_nonvap[-1]
        message = f"L'eau qui entre dans la chaudiere contient de la vapeur pour mdot >= {m_dot[imdot_aj_sup]} "

elif vapeur_cond:
    imdot_aj_sup = index_titre_cond
    message = f"L'eau en sortie du condenseur est entierement sous forme de vapeur pour mdot >= {m_dot[imdot_aj_sup]} "

elif vapeur_chaud:
    imdot_aj_inf = int_nonvap[0]
    imdot_aj_sup = int_nonvap[-1]
    if int_nonvap[-1] == 0:
        int_sup = len(m_dot) - 1
    else:
        int_sup = imdot_aj_sup

    message = f"L'eau qui entre dans la chaudiere ne contient pas de vapeur pour {m_dot[imdot_aj_inf]} < mdot < {round(m_dot[int_sup],2)} "
else:
    message = ''

if vapeur_cond or vapeur_chaud:
    m_dot = m_dot[imdot_aj_inf:imdot_aj_sup]
    W_net = W_net[imdot_aj_inf:imdot_aj_sup]
    rendement = rendement[imdot_aj_inf:imdot_aj_sup]
    eau_alim = eau_alim[imdot_aj_inf:imdot_aj_sup]
    sheet.Range("A17").Value = message

dim_mdot = np.size(m_dot)
suite = True
if dim_mdot == 0:
    sheet.Range("A18").Value = "Les calculs n'ont pas pu etre faits, aucun point n'est valide"
    suite = False
elif dim_mdot == 1:
    p_opt = m_dot[0]
else:
    opt_inf = data['debit'][4]
    opt_sup = data['debit'][5]

    if opt_inf not in m_dot:
        opt_ancien = opt_inf
        opt_inf = m_dot[0]
        message = f"Le point {opt_ancien} est un point ou l'eau en entierement en vapeur, le point a ete remplace par {opt_inf}"
        sheet.Range("M9").Value = message
        # TODO higlight

    if opt_sup not in m_dot:
        opt_ancien = opt_inf
        opt_sup = m_dot[-1]
        message = f"Le point {opt_ancien} est un point ou l'eau en entierement en vapeur, le point a ete remplace par {opt_sup}"
        sheet.Range("M10").Value = message
        # TODO highlight

    p_opt = math_util.point_optimal(m_dot, rendement, opt_inf, opt_sup)

if suite:
    if isinstance(p_opt, str):
        sheet.Range("B13").Value = p_opt
    else:
        rend_opt = np.interp(p_opt, m_dot, rendement)
        W_net_opt = np.interp(p_opt, m_dot, W_net)
        sheet.Range("B13").Value = round(W_net_opt/1000, 4)
        sheet.Range("B14").Value = round(p_opt, 2)
        sheet.Range("B15").Value = round(rend_opt, 4)

    # TODO controler la grosseur du graph, mais scale down pour excel
    # TODO transformer en fonction
    plt.figure(dpi=100)
    titre_figa = titre_sim + 'a'
    plt.title('Rendement selon la puissance\n'+f'simulation {titre_sim}')
    plt.plot(W_net, rendement)
    path_figa = os.path.join(titre_sim, titre_figa)
    plt.savefig(path_figa)
    fig_cell = sheet.Range("A20")

    fig = sheet.Pictures().Insert(path_figa+'.png')
    fig.Left = fig_cell.Left
    fig.Top = fig_cell.Top

    plt.figure(dpi=100)
    titre_figb = titre_sim + 'b'
    plt.title('Rendement selon le debit massique\n'+f'simulation {titre_sim}')
    plt.plot(m_dot, rendement)
    path_figb = os.path.join(titre_sim, titre_figb)
    plt.savefig(path_figb)
    fig_cell = sheet.Range("A45")

    fig = sheet.Pictures().Insert(path_figb+'.png')
    fig.Left = fig_cell.Left
    fig.Top = fig_cell.Top

    plt.figure(dpi=100)
    titre_figc = titre_sim + 'c'
    plt.title('Rendement selon eau alim\n'+f'simulation {titre_sim}')
    plt.plot(W_net, eau_alim)
    path_figc = os.path.join(titre_sim, titre_figc)
    plt.savefig(path_figc)
    fig_cell = sheet.Range("A70")

    fig = sheet.Pictures().Insert(path_figc+'.png')
    fig.Left = fig_cell.Left
    fig.Top = fig_cell.Top

sheet.Protect()
