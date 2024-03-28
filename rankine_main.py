import time
start = time.time()
print("Démarrage de la simulation...")
import traceback


import os
from rankine_fct import *
import win32com.client
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fexcel
import csv
import shutil


class Espacetitre(Exception):
    def __init__(self, notice=None):
        self.message = notice
        super().__init__(notice)


print('Lecture des donnees')
file_path = "interface.xlsm"
data = pd.read_excel(file_path, header=0, sheet_name='Données')
BD = pd.read_excel(file_path, header=0, sheet_name="Base de données")

inf = data['Centrale'][12]
sup = data['Centrale'][13]
n_pts = int(data['Centrale'][14])

m_dot = np.linspace(inf, sup, n_pts)

ExcelApp = win32com.client.GetActiveObject("Excel.Application")
ExcelApp.Visible = True

workbook = ExcelApp.Workbooks.Open(r"interface.xlsm")
sheet = workbook.Worksheets(2)

titre_sim = f'{sheet.Range("B2").Value}'

cell_erreur = "D4"
cell_message1 = "D5"
cell_message2 = "D6"
cell_message3 = "D7"
cell_message4 = "D8"

if " " in titre_sim:
    sheet.Range(cell_erreur).Interior.ColorIndex = 3
    sheet.Range(cell_erreur).Value = "Erreur"
    sheet.Range(cell_message1).Font.ColorIndex = 3
    sheet.Range(cell_message1).Value = "Le titre de la simulation contient un espace"
    sheet.Range(cell_message2).Font.ColorIndex = 3
    sheet.Range(cell_message2).Value = "Les calculs n'ont pas été faits"
    raise Espacetitre(
        "Le titre de la simulation contient un espace"
    )

try:
    os.mkdir(titre_sim)
except FileExistsError:
    pass
shutil.copy('interp.py', f'{titre_sim}')

try:
    W_net, rendement, eau_alim, w_p_lp, w_p_hp, temps_surdemande, emissions, vapeur_cond, index_titre_cond,\
        vapeur_chaud, int_nonvap, epaisseur, diametre, T_riv, t_graph, s_graph = travail(m_dot, data, BD)
    W_net = W_net / 1000  # MW
    print('Traitement')

    rend_hp = rendement[0]
    rend_lp = rendement[1]
    rend_cycle = rendement[2]

    imdot_aj_inf = 0
    imdot_aj_sup = len(m_dot) - 1

    message = ''
    if vapeur_cond and vapeur_chaud:
        imdot_aj_inf = int_nonvap[0]
        imdot_aj_sup = int_nonvap[-1]
        if index_titre_cond < int_nonvap[-1]:
            imdot_aj_sup = index_titre_cond
            message = f"L'eau qui sort du condenseur est entierement en phase vapeur pour mdot >= {m_dot[imdot_aj_sup]}"

    elif vapeur_cond:
        imdot_aj_sup = index_titre_cond
        message = f"L'eau en sortie du condenseur est entierement sous forme de vapeur pour mdot " \
                  f">= {m_dot[imdot_aj_sup]} "

    elif vapeur_chaud:
        imdot_aj_inf = int_nonvap[0]
        imdot_aj_sup = int_nonvap[-1]
        message = f"L'eau qui entre dans la chaudiere respecte la tolérance de teneur en vapeur pour  {round(m_dot[imdot_aj_inf], 2)} < mdot < " \
                  f"{round(m_dot[imdot_aj_sup], 2)} "

    if vapeur_cond or vapeur_chaud:
        imdot_aj_sup += 1
        m_dot = m_dot[imdot_aj_inf:imdot_aj_sup]
        W_net = W_net[imdot_aj_inf:imdot_aj_sup]

        rend_hp = rendement[0][imdot_aj_inf:imdot_aj_sup]
        rend_lp = rendement[1][imdot_aj_inf:imdot_aj_sup]
        rend_cycle = rendement[2][imdot_aj_inf:imdot_aj_sup]
        eau_alim = eau_alim[imdot_aj_inf:imdot_aj_sup]
        w_p_lp = w_p_lp[imdot_aj_inf:imdot_aj_sup]
        w_p_hp = w_p_hp[imdot_aj_inf:imdot_aj_sup]
        emissions = emissions[imdot_aj_inf:imdot_aj_sup]
        temps_surdemande = temps_surdemande[imdot_aj_inf:imdot_aj_sup]
        T_riv = T_riv[imdot_aj_inf:imdot_aj_sup]

        sheet.Range(cell_erreur).Interior.ColorIndex = 3
        sheet.Range(cell_erreur).Value = 'Avertissement'
        sheet.Range(cell_message2).Font.ColorIndex = 3
        sheet.Range(cell_message2).Value = message
        sheet.Range(cell_message3).Value = f"Les calculs ont été fais pour {round(m_dot[0],2)} < mdot < " \
                                           f"{round(m_dot[-1], 2)} "

    dim_mdot = np.size(m_dot)
    suite = True
    if dim_mdot == 0:
        sheet.Range(cell_erreur).Interior.ColorIndex = 3
        sheet.Range(cell_erreur).Value = "Erreur"
        sheet.Range(cell_message1).Value = "Les calculs n'ont pas pu etre faits, aucun point n'est valide"
        sheet.Range(cell_message1).Font.ColorIndex = 3
        p_opt = 0
        suite = False
    elif dim_mdot == 1:
        p_opt = m_dot[0]
        sheet.Range(cell_message1).Value = f"Un seul point est considéré dans les calculs, soit {round(p_opt)} "
    else:
        opt_inf = data['Centrale'][15]
        opt_sup = data['Centrale'][16]

        if opt_inf < m_dot[0]:
            opt_ancien = opt_inf
            opt_inf = m_dot[0]
            message = f"Optimisation: Le point {round(opt_ancien)} " \
                      f"est un point ou l'eau en entierement en vapeur, le point a ete remplace par {round(opt_inf)}"
            sheet.Range(cell_message3).Value = message
            sheet.Range(cell_message3).Font.ColorIndex = 3

        if opt_sup > m_dot[-1]:
            opt_ancien = opt_sup
            opt_sup = m_dot[-1]
            message = f"Optimisation: Le point {round(opt_ancien)} " \
                      f"est un point ou l'eau en entierement en vapeur, le point a ete remplace par {round(opt_sup)}"
            sheet.Range(cell_message4).Value = message
            sheet.Range(cell_message4).Font.ColorIndex = 3

        p_opt = math_util.point_optimal(m_dot, rend_cycle, opt_inf, opt_sup)

    index_temps = len(m_dot) - 1
    if suite:
        if isinstance(p_opt, str):
            sheet.Range(cell_message1).Value = p_opt
        else:
            cap_evap = data["Centrale"][21]
            index_temps = len(m_dot) - 1
            for i, m in enumerate(m_dot):
                if m > cap_evap:
                    index_temps = i
                    break

            rend_opt = np.interp(p_opt, m_dot, rend_cycle)
            W_net_opt = np.interp(p_opt, m_dot, W_net)

            sheet.Range("A5").Value = "Puissance"
            sheet.Range("C5").Value = "MW"
            sheet.Range("A6").Value = "Debit"
            sheet.Range("C6").Value = "kg/s"
            sheet.Range("A7").Value = "Rendement"

            sheet.Range("B5").Value = round(W_net_opt, 4)
            sheet.Range("B6").Value = round(p_opt, 2)
            sheet.Range("B7").Value = round(rend_opt, 4)

        i = 0
        sheet.Range("B9").Value = "Diamètre interne de la tuyauterie (mm)"
        sheet.Range("B9").BorderAround(LineStyle=1, ColorIndex=51)
        sheet.Range("C9").Value = "Épaisseur de tuyau (mm)"
        sheet.Range("C9").BorderAround(LineStyle=1, ColorIndex=51)
        for i, t in enumerate(epaisseur):
            index_etage = str(10 + i)
            sheet.Range("A" + index_etage).Value = f'Étage {i + 1}'
            sheet.Range("A" + index_etage).BorderAround(LineStyle=1, ColorIndex=51)
            if t == 0:
                epais = ""
                dia = "Pas de resurchauffe"
            else:
                dia = round(diametre[i]*1000)
                epais = round(t*1000)
            sheet.Range("B" + index_etage).Value = dia
            sheet.Range("C" + index_etage).Value = epais
            sheet.Range("B" + index_etage).BorderAround(LineStyle=1, ColorIndex=51)
            sheet.Range("C" + index_etage).BorderAround(LineStyle=1, ColorIndex=51)

        index_tableau = 12 + i

        fexcel.graphique(150, titre_sim, 'Rendement selon la puissance', 'Puissance [MW]', 'Rendement', 'a', W_net,
                         rend_cycle, sheet, "A" + str(index_tableau))

        fexcel.graphique(150, titre_sim, 'Rendement selon le debit', 'Débit [kg/s]', 'Rendement', 'b', m_dot, rend_cycle,
                         sheet, "A" + str(index_tableau + 25))

        fexcel.graphique(150, titre_sim, 'Débit d\'eau d\'alimentation selon la puissance', 'Puissance [MW]',
                         'Débit d\'eau d\'alimentation [kg/s]', 'c', W_net,
                         eau_alim, sheet, "A" + str(index_tableau + 50))

        fexcel.graphique(150, titre_sim, 'Puissance de la pompe basse pression', 'Puissance [MW]', 'Puissance [kW]',
                         'd', W_net, w_p_lp, sheet,
                         "A" + str(index_tableau + 75))

        fexcel.graphique(150, titre_sim, 'Puissance de la pompe haute pression', 'Puissance [MW]', 'Puissance [kW]',
                         'e', W_net, w_p_hp, sheet,
                         "A" + str(index_tableau + 100))

        fexcel.graphique(150, titre_sim, 'Temps de surdemande selon la puissance', 'Puissance [MW]', 'Temps [heures]',
                         'f', W_net[index_temps:],
                         temps_surdemande[index_temps:], sheet, "A" + str(index_tableau + 125))

        fexcel.graphique(150, titre_sim, 'Émissions de CO2 selon la puissance', 'Puissance [MW]',
                         'Émissions [kgCO2/kWh]', 'g', W_net, emissions, sheet,
                         "A" + str(index_tableau + 150))

        fexcel.graphrendturb(150, titre_sim, "Rendement des turbines", "Débit [kg/s]", "Rendement", 'h', m_dot,
                             [rend_hp, rend_lp], sheet, "H41")

        fexcel.graphique(150, titre_sim, "Température de rejet dans la rivière selon la puissance produite",
                         "Puissance [MW]", "Température [K]", 'i', W_net, T_riv, sheet, "A" + str(index_tableau + 175))

        point_TS = data['Centrale'][18] - 1
        plt.figure(dpi=150)
        plt.plot(s_graph, t_graph)
        T_liq = np.linspace(270, 647, 100)
        s_cloche1 = H2O.s(x=0, T=T_liq)
        T_vap = np.linspace(270, 647, 100)
        s_cloche2 = H2O.s(x=1, T=T_vap)
        plt.plot(s_cloche1, T_liq, color='black')
        plt.plot(s_cloche2, T_vap, color='black')
        plt.title(f"Diagramme T-S de l'eau à $\\dot{{m}} = $ {round(m_dot[point_TS])}kg/s \n "
                  f"Simulation {titre_sim}")
        plt.ylabel("Température [K]")
        plt.xlabel("Entropie [kJ/(kgK)]")
        for i in range(len(t_graph)-1):
            label = str(i + 2)
            plt.annotate(label, (s_graph[i], t_graph[i]))
        nom_fichier = titre_sim + "j"
        path_fig = os.path.join(titre_sim, nom_fichier )
        plt.savefig(path_fig)
        fig_cell = sheet.Range("A" + str(index_tableau + 200))

        fig = sheet.Pictures().Insert(path_fig + '.png')
        fig.Height = 350
        fig.Left = fig_cell.Left
        fig.Top = fig_cell.Top

        path = titre_sim + '/' + 'donneinterp.csv'
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(titre_sim)
            writer.writerow(W_net)
            writer.writerow(rend_cycle)
        sheet.Range("A3").Value = 'Temps de calcul (sec):'
        sheet.Range("B3").Value = round((time.time() - start))

except SoutirageInvalide:
    sheet.Range(cell_erreur).Interior.ColorIndex = 3
    sheet.Range(cell_erreur).Value = "Erreur"
    sheet.Range(cell_message1).Font.ColorIndex = 3
    sheet.Range(cell_message1).Value = "Le soutirage n'est pas valide"

except Resurchauffe as e:
    sheet.Range(cell_erreur).Interior.ColorIndex = 3
    sheet.Range(cell_erreur).Value = "Erreur"
    sheet.Range(cell_message1).Font.ColorIndex = 3
    sheet.Range(cell_message1).Value = str(e)

except VapeurChaudiere as e:
    sheet.Range(cell_erreur).Interior.ColorIndex = 3
    sheet.Range(cell_erreur).Value = "Erreur"
    sheet.Range(cell_message1).Font.ColorIndex = 3
    sheet.Range(cell_message1).Value = str(e)
    sheet.Range(cell_message2).Font.ColorIndex = 3
    sheet.Range(cell_message2).Value = "Les calculs n'ont pas été faits"

except VapeurCond as e:
    sheet.Range(cell_erreur).Interior.ColorIndex = 3
    sheet.Range(cell_erreur).Value = "Erreur"
    sheet.Range(cell_message1).Font.ColorIndex = 3
    sheet.Range(cell_message1).Value = str(e)
    sheet.Range(cell_message2).Font.ColorIndex = 3
    sheet.Range(cell_message2).Value = "Les calculs n'ont pas été faits"

except Exception as e:
    sheet.Range(cell_erreur).Interior.ColorIndex = 3
    sheet.Range(cell_erreur).Value = "Erreur"
    sheet.Range(cell_message1).Font.ColorIndex = 3
    sheet.Range(cell_message1).Value = str(e)
    sheet.Range(cell_message2).Value = traceback.format_exc()

finally:
    sheet.Cells.Locked = False
    sheet.Range("A:AA").Locked = True
    sheet.Range("E2:E3").Locked = False
    sheet.Protect()
