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
data = pd.read_excel(file_path, header=0)

# h, p = rankine_fct.turb_it1(data)
# mdot = data['turbine'][0]
# Sy = data['centrale'][4]
# FS = data['centrale'][5]
# dia, v = pertes.diametre(mdot, p, h)
# t = pertes.epaisseur(p, dia, Sy, FS)
# dQ, dP = pertes.dQdP(h, p, data, dia, v, t)
# print(dQ, dP)


print(rankine_fct.d_v_t(data))

"""la valeur de Ts change tres peu les pertes de chaleur on peut prendre Ts = T vap"""
# Ts = np.linspace(320, 750, 100)
# Q = np.zeros(np.size(Ts))
#
# for i, T in enumerate(Ts):
#     R_c_int = pertes.R_conv(mdot, dia, h, p, L)
#     R_tuyau = pertes.R_cond(dia, L, t, k_acier)
#     R_iso = pertes.R_cond(dia+2*t, L, t_iso, k_iso)
#     R_nat = pertes.R_conv_nat(T, Tinf, dia+2*t+2*t_iso, L)
#
#     R_tot = R_c_int + R_tuyau + R_iso + R_nat
#     Q[i] = (600 - 300) / R_tot / 1000
#
# plt.plot(Ts, Q)
# plt.show()


