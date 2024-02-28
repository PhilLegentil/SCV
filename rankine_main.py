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

W_net, rendement, eau_alim = travail(m_dot, data)
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

opt_inf = data['debit'][4]
opt_sup = data['debit'][5]
p_opt = math_util.point_optimal(m_dot, rendement, opt_inf, opt_sup)

#  TODO donner des indications pour quand point_optimal ne donne pas de zero mais que la courbe a clairement un optimum
if isinstance(p_opt, str):
    sheet.Range("B13").Value = p_opt
else:
    rend_opt = np.interp(p_opt, m_dot, rendement)
    W_net_opt = np.interp(p_opt, m_dot, W_net)
    sheet.Range("B13").Value = round(W_net_opt/1000, 4)
    sheet.Range("B14").Value = round(p_opt, 2)
    sheet.Range("B15").Value = round(rend_opt, 4)

# TODO controler la grosseur du graph, mais scale down pour excel
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
