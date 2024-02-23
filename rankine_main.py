print("Démarrage de la simulation...")

import os
from rankine_fct import *
import win32com.client
import matplotlib.pyplot as plt
import numpy as np
import math_util as mut
import pandas as pd

print('Calcul...')
file_path = "interface.xlsm"
data = pd.read_excel(file_path, header=0)

inf = data['debit'][0]
sup = data['debit'][1]
n_pts = int(data['debit'][2])

m_dot = np.linspace(inf, sup, n_pts)

W_net, rendement = travail(m_dot, data)
print('Traitement')

p_opt = mut.point_optimal(m_dot, rendement, inf, sup)
rend_opt = mut.fonction_poly(m_dot, rendement, p_opt)
W_net_opt = mut.fonction_poly(m_dot, W_net, p_opt)

titre_sim = f'{data["noms"][0]}'

try:
    os.mkdir(titre_sim)
except:
    pass

plt.figure(dpi=100)
titre_figa = titre_sim + 'a'
plt.title('Rendement selon la puissance\n'+f'simulation {titre_sim}')
plt.plot(W_net, rendement)
path_figa = os.path.join(titre_sim, titre_figa)
plt.savefig(path_figa)

ExcelApp = win32com.client.GetActiveObject("Excel.Application")
ExcelApp.Visible = True

workbook = ExcelApp.Workbooks.Open(r"interface.xlsm")


sheet = workbook.Worksheets(2)
sheet.Range("B13").Value = round(W_net_opt/1000, 4)
sheet.Range("B14").Value = round(p_opt, 2)
sheet.Range("B15").Value = round(rend_opt, 4)


figa_cell = sheet.Range("H14")

figa = sheet.Pictures().Insert(path_figa+'.png')
figa.Left = figa_cell.Left
figa.Top = figa_cell.Top

sheet.Protect()
