print("Démarrage de la simulation...")

import os
from rankine_fct import *
import win32com.client
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fexcel
import csv
import shutil

print('Lecture des donnees')
file_path = "interface.xlsm"
data = pd.read_excel(file_path, header=0, sheet_name='Données')
BD = pd.read_excel(file_path, header=0, sheet_name="Base de données")

inf = data['debit'][0]
sup = data['debit'][1]
n_pts = int(data['debit'][2])

m_dot = np.linspace(inf, sup, n_pts)

ExcelApp = win32com.client.GetActiveObject("Excel.Application")
ExcelApp.Visible = True

workbook = ExcelApp.Workbooks.Open(r"interface.xlsm")
sheet = workbook.Worksheets(2)

titre_sim = f'{sheet.Range("A18").Value}'

try:
    os.mkdir(titre_sim)
except FileExistsError:
    pass
shutil.copy('interp.py', f'{titre_sim}')

try:
    W_net, rendement, eau_alim, w_p_lp, w_p_hp, temps_surdemande, index_temps, emissions, vapeur_cond, index_titre_cond, vapeur_chaud, int_nonvap, epaisseur = travail(m_dot, data, BD)
    W_net = W_net/1000  # MW
    print('Traitement')

    imdot_aj_inf = 0
    imdot_aj_sup = len(m_dot) - 1

    # TODO fix the logic
    if vapeur_cond and vapeur_chaud:

        if index_titre_cond < int_nonvap[-1]:
            imdot_aj_sup = index_titre_cond
            message = f"L'eau qui sort du condenseur est entierement en phase vapeur pour mdot >= {m_dot[imdot_aj_sup]}"
        else:
            imdot_aj_sup = int_nonvap[-1]
            message = f"L'eau qui entre dans la chaudiere contient de la vapeur pour mdot >= {m_dot[imdot_aj_sup]} "
        imdot_aj_inf = int_nonvap[0]
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

        message = f"L'eau qui entre dans la chaudiere contient trop de vapeur pour {m_dot[imdot_aj_inf]} < mdot < {round(m_dot[int_sup],2)} "
    else:
        message = ''

    if vapeur_cond or vapeur_chaud:
        m_dot = m_dot[imdot_aj_inf:imdot_aj_sup]
        W_net = W_net[imdot_aj_inf:imdot_aj_sup]
        rendement = rendement[imdot_aj_inf:imdot_aj_sup]
        eau_alim = eau_alim[imdot_aj_inf:imdot_aj_sup]
        sheet.Range("A15").Interior.ColorIndex = 3
        sheet.Range("A17").Font.ColorIndex = 3
        sheet.Range("A17").Value = message

    dim_mdot = np.size(m_dot)
    suite = True
    if dim_mdot == 0:
        sheet.Range("A15").Interior.ColorIndex = 3
        sheet.Range("A15").Value = "Erreur"
        sheet.Range("A16").Value = "Les calculs n'ont pas pu etre faits, aucun point n'est valide"
        sheet.Range("A16").Font.ColorIndex = 3
        suite = False
    elif dim_mdot == 1:
        p_opt = m_dot[0]
    else:
        opt_inf = data['debit'][4]
        opt_sup = data['debit'][5]

        if opt_inf < m_dot[0]:
            opt_ancien = opt_inf
            opt_inf = m_dot[0]
            message = f"Le point {opt_ancien} est un point ou l'eau en entierement en vapeur, le point a ete remplace par {opt_inf}"
            sheet.Range("M9").Value = message
            sheet.Range("M9").Font.ColorIndex = 3

        if opt_sup > m_dot[-1]:
            opt_ancien = opt_inf
            opt_sup = m_dot[-1]
            message = f"Le point {opt_ancien} est un point ou l'eau en entierement en vapeur, le point a ete remplace par {opt_sup}"
            sheet.Range("M10").Value = message
            sheet.Range("M10").Font.ColorIndex = 3

        p_opt = math_util.point_optimal(m_dot, rendement, opt_inf, opt_sup)

    if suite:
        if isinstance(p_opt, str):
            sheet.Range("B15").Value = p_opt
        else:
            rend_opt = np.interp(p_opt, m_dot, rendement)
            W_net_opt = np.interp(p_opt, m_dot, W_net)

            sheet.Range("A15").Value = "Puissance"
            sheet.Range("C15").Value = "MW"
            sheet.Range("A16").Value = "Debit"
            sheet.Range("C16").Value = "kg/s"
            sheet.Range("A17").Value = "Rendement"
            sheet.Range("A19").Value = "Courbe de rendement"

            sheet.Range("B15").Value = round(W_net_opt, 4)
            sheet.Range("B16").Value = round(p_opt, 2)
            sheet.Range("B17").Value = round(rend_opt, 4)

        fexcel.graphique(150, titre_sim, 'Rendement selon la puissance', 'Puissance', 'Rendement', 'a', W_net, rendement, sheet, "A22")

        fexcel.graphique(150, titre_sim, 'Rendement selon le debit', 'Debit', 'Rendement', 'b', m_dot, rendement, sheet, 'A47')

        fexcel.graphique(150, titre_sim, 'Eau alim selon la puissance', 'Puissance', 'Debit eau alim', 'c', W_net, eau_alim, sheet, 'A72')

        fexcel.graphique(150, titre_sim, 'Pompe LP', 'Puissance', 'Puissance', 'd', W_net, w_p_lp, sheet, 'A97')

        fexcel.graphique(150, titre_sim, 'Pompe HP', 'Puissance', 'Puissance', 'e', W_net, w_p_hp, sheet, 'A122')

        fexcel.graphique(150, titre_sim, 'Temps surdemande', 'Puissance', 'Temps', 'f', W_net[index_temps:], temps_surdemande, sheet, 'A147')

        fexcel.graphique(150, titre_sim, 'Emissions', 'Puissance', 'Emissions', 'g', W_net, emissions, sheet, 'A172')

        for i, t in enumerate(epaisseur):
            sheet.Range("A" + str(198 + i)).Value = f'Étage {i+1}'
            if t == 0:
                value = 'Pas de resurchauffe'
            else:
                value = t
            sheet.Range("B" + str(198+i)).Value = value

        path = titre_sim + '/' + 'donneinterp.csv'
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(titre_sim)
            writer.writerow(W_net)
            writer.writerow(rendement)


except SoutirageInvalide:
    sheet.Range("A15").Interior.ColorIndex = 3
    sheet.Range("A15").Value = "Erreur"
    sheet.Range("A16").Font.ColorIndex = 3
    sheet.Range("A16").Value = "Le soutirage n'est pas valide"

except Resurchauffe as e:
    sheet.Range("A15").Interior.ColorIndex = 3
    sheet.Range("A15").Value = "Erreur"
    sheet.Range("A16").Font.ColorIndex = 3
    sheet.Range("A16").Value = str(e)

sheet.Cells.Locked = False
sheet.Range("A:W").Locked = True
sheet.Protect()
