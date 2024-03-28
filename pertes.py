import pyromat as pm
import numpy as np
import pyfluids as pf

import matplotlib.pyplot as plt


'''hypothese d'ecoulement pleinement developpe'''

pm.config['unit_pressure'] = 'kPa'
pm.config['unit_energy'] = 'kJ'
H2O = pm.get("mp.H2O")


def diametre(m_dot, p, h):
    v = 0
    if p >= 15200:
        v = 35
    if 5066 <= p < 15200:
        v = 40
    if 2026 <= p < 5066:
        v = 45
    if p < 2026:
        v = 25

    rho = 1 / H2O.v(p=p, h=h)
    dia = np.sqrt(m_dot * 4 / (rho * np.pi * v))

    return dia, v  # metres et m/s


def epaisseur(p, dia, Sy, FS):
    Sy = Sy * 1000  # conversion MPa a kPa
    t = FS * p * dia / 2 / Sy
    return t


def R_conv(mdot, dia, h, p, L):
    vapeur = pf.Fluid(pf.FluidsList.Water).with_state(
        pf.Input.pressure(p * 1e3), pf.Input.enthalpy(h * 1e3)  # en Pa et J
    )
    mu = vapeur.dynamic_viscosity
    Pr = vapeur.prandtl
    k = vapeur.conductivity
    Re = 4 * mdot / (np.pi * dia * mu)
    Nu = 0.023 * Re ** (4 / 5) * Pr ** 0.3  # Ts < Tm

    h_int = Nu * k / dia

    perim = 2 * np.pi * (dia / 2)
    R = 1 / (h_int * perim * L)
    return R, Re


def R_cond(dia, L, t, k):
    R = np.log((dia + t) / dia) / (2 * np.pi * L * k)

    return R


def R_conv_nat(Ts, Tinf, dia, L):
    p = 100000
    g = 9.81
    Tm = (Ts + Tinf) / 2
    air = pf.Fluid(pf.FluidsList.Air).with_state(
        pf.Input.pressure(p * 1e3), pf.Input.temperature(Tm - 273.15)  # en Pa et degC
    )
    beta = 1 / Tm
    nu = air.kinematic_viscosity
    k = air.conductivity
    rho = air.density
    c = air.specific_heat
    Pr = air.prandtl
    alpha = k / (rho * c)

    Ra = g * beta * (Ts - Tinf) * dia ** 3 / (nu * alpha)

    Nu = (0.6 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2

    h_int = Nu * k / dia

    perim = 2 * np.pi * (dia / 2)
    R = 1 / (h_int * perim * L)

    return R, h_int


def p_chaleur(mdot, dia, h, p, k_acier, k_iso, L, t_iso, t):
    R_c_int = R_conv(mdot, dia, h, p, L)
    R_tuyau = R_cond(dia, L, t, k_acier)
    R_iso = R_cond(dia + 2 * t, L, t_iso, k_iso)
    R_nat = R_conv_nat(330, 300, dia + 2 * t + 2 * t_iso, L)

    R_tot = R_c_int + R_tuyau + R_iso + R_nat
    Q = (600 - 300) / R_tot / 1000

    return Q


def p_charge(v, Re, rug, dia, L, h, p):
    rug_rel = rug / dia
    A = (rug_rel / 3.7) ** 1.11
    B = 6.9 / Re
    f = (1 / (-1.8 * np.log10(A + B))) ** 2

    rho = 1 / H2O.v(h=h, p=p)

    dP = f * L * rho * v ** 2 / (2 * dia)

    return dP / 1000  # kPa


def dQdP(h, h_prop, mdot, p, data, BD, dia, v, t):
    mat = BD['Choixiso'][0]
    index_iso = BD.index[BD['Isolant'] == mat].tolist()[0]
    k_iso = BD['Conductivité de l\'isolant [W/(mK)]'][index_iso]

    tu = BD['Choixtu'][0]
    index_tu = BD.index[BD['Métal'] == tu].tolist()[0]
    rugosite = BD['Rugosité'][index_tu]
    k_acier = BD['Conductivité du métal [W/(mK)]'][index_tu]

    # mdot = data['Turbines'][0]
    L = data['Centrale'][4]

    # t_iso = np.linspace(0.0001,5, 50)
    t_iso = data['Centrale'][6]
    Tinf = data['Centrale'][7]

    Tvap = H2O.T(h=h, p=p)
    Ts = H2O.T(h=h_prop, p=p)

    R_c_int, Re = R_conv(mdot, dia, h_prop, p, L)
    R_tuyau = R_cond(dia, L, t, k_acier)
    R_iso = R_cond(dia + 2 * t, L, t_iso, k_iso)
    R_nat, h_nat = R_conv_nat(Ts, Tinf, dia + 2 * t + 2 * t_iso, L)
    R_tot = R_c_int + R_tuyau + R_iso + R_nat
    UA = 1 / R_tot
    cp = H2O.cp(h=h, p=p)
    dTi = (Tinf - Tvap)
    dTo = np.exp(-UA / (mdot * cp)) * dTi

    dTlm = (dTo - dTi) / np.log(dTo / dTi)

    dQ = UA * dTlm / 1000
    # plt.plot(t_iso, k_iso/h_nat)
    # plt.title("Rayon critique selon l'épaisseur d'isolant")
    # plt.xlabel("Épaisseur [m]")
    # plt.ylabel("Rayon critique [m]")
    # plt.show()

    # p_opt = math_util.point_optimal(t_iso, dQ, 0.0001, 5)
    # print(p_opt)

    dP = p_charge(v, Re, rugosite, dia, L, h, p)

    return dQ, dP  # kW (dQ negatif pour une perte) et kPa (toujours positif)
