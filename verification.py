import math_util as mut
import pertes
import pandas as pd
import rankine_fct
import numpy as np
import matplotlib.pyplot as plt
import pyromat as pm

"""verification calcul fractions"""
fractions = [0.2, 0.3, 0.4]
deb_turb = mut.frac_mdot_turbine(fractions, 2)
deb_sout = 0
for i in range(0, len(fractions)):
    deb_sout += mut.frac_mdot_sout(fractions, i)
    print('deb', deb_sout)
if deb_sout + deb_turb == 1:
    print('valide')
else:
    print('non valide')

pm.config['unit_pressure'] = 'kPa'
pm.config['unit_energy'] = 'kJ'
H2O = pm.get("mp.H2O")

file_path = "interface.xlsm"
data = pd.read_excel(file_path, header=0, sheet_name="Données")
BD = pd.read_excel(file_path, header=0, sheet_name="Base de données")


"""la valeur de Ts change tres peu les pertes de chaleur on peut prendre Ts = T vap"""

mdot = data['Turbines'][0]
dia, v, t, h_out = rankine_fct.d_v_t(data,BD)

h = h_out[0]
dia = dia[0]
v = v[0]
t = t[0]

p = data['Turbines'][5]
L = data['Centrale'][4]

mat = BD['Choixiso'][0]
index_iso = BD.index[BD['Isolant'] == mat].tolist()[0]
k_iso = BD['Conductivité de l\'isolant [W/(mK)]'][index_iso]

tu = BD['Choixtu'][0]
index_tu = BD.index[BD['Métal'] == tu].tolist()[0]
rugosite = BD['Rugosité'][index_tu]
k_acier = BD['Conductivité du métal [W/(mK)]'][index_tu]

t_iso = data['Centrale'][6]
Tinf = data['Centrale'][7]

Tmaxturb = data['Turbines'][4]

Ts = np.linspace(Tinf + 10, Tmaxturb, 100)
dQ = np.zeros(np.size(Ts))

R_tot = 0
for i, T in enumerate(Ts):
    R_c_int, Re = pertes.R_conv(mdot, dia, h, p, L)
    R_tuyau = pertes.R_cond(dia, L, t, k_acier)
    R_iso = pertes.R_cond(dia+2*t, L, t_iso, k_iso)
    R_nat = pertes.R_conv_nat(T, Tinf, dia+2*t+2*t_iso, L)

    R_tot = R_c_int + R_tuyau + R_iso + R_nat

    UA = 1 / R_tot
    cp = H2O.cp(h=h, p=p)
    dTi = (Tinf - Tmaxturb)
    dTo = np.exp(-UA / (mdot * cp)) * dTi

    dTlm = (dTo - dTi) / np.log(dTo / dTi)

    dQ[i] = UA * dTlm / 1000


plt.plot(Ts, dQ)
plt.title("Pertes de chaleur selon la température à la surface de l'isolant \n"
          "Pour $T_{inf} + 10 \\leq T_s \\leq < T_{vap}$")
plt.xlabel("Température [K]")
plt.ylabel("Perte de chaleur [kW]")
plt.show()

