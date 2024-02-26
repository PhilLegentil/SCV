import pyromat as pm
import numpy as np
import scipy as sp
import math_util as mut

pm.config['unit_pressure'] = 'kPa'
pm.config['unit_energy'] = 'kJ'
H2O = pm.get("mp.H2O")


def fig_6(p_in, p_out, V_dot):
    c = np.array([[11.151, -63], [-0.50091, 2.83]])
    x = p_out / p_in
    y = np.log(V_dot)
    perte = mut.biv_poly(x, y, c)

    return perte[0] / 100


def fig_7(PD):
    PD = PD
    coef = [-0.115, 4.37]
    delta = np.polyval(coef, PD)

    return delta / 100


def fig_8(TFR, PD):
    coef = np.array([[-21.8085, 21.8085], [0.573908, -0.573908]])

    if isinstance(TFR, float):
        x = TFR
        y = PD
        perte = mut.biv_poly(x, y, coef)
        return perte/100
    else:
        x = np.array([TFR])
        y = np.full((1, len(TFR)), PD)
        perte = mut.biv_poly(x, y, coef)
        return perte[0] / 100


def fig_9(p_in, p_out, TFR):
    coef = np.array([[-60.75, 66.85, 29.75, -35.85],
                     [17.5, -20.02, -0.525, 3.045]])
    r = p_in / p_out

    if isinstance(TFR, float):
        x = TFR
        y = np.log(r)
        perte = mut.biv_poly(x, y, coef)
        return perte/100
    else:
        x = np.array([TFR])
        y = np.full((1, len(TFR)), np.log(r))
        perte = mut.biv_poly(x, y, coef)
        return perte[0] / 100


def fig_14(p_in, h_in):
    coef = np.array([
        [28.232252, -92.390491, -625.79590, 207.23010, 70.251642, -22.516388],
        [-0.047796308, 1.2844571, 0.3855961, -0.039652999, -0.27180357, 0.064869467],
        [-0.69791427e-3, -0.17037268e-2, 0.86563845e-3, -0.5951066e-3, 0.39705804e-3, -0.73533255e-4],
        [0.12050837e-5, 0.26826382e-6, -0.67887771e-6, 0.52886157e-6, -0.24106229e-6, 0.37881801e-7],
        [-0.50719109e-9, 0.26393497e-9, 0.38021911e-10, -0.10149993e-9, 0.47757232e-10, -0.70989561e-11]
    ])
    x = np.log10(p_in * 0.145038)  # kpa to psia
    y = h_in * 0.429923  # kj/kg to btu/lb

    perte = mut.biv_poly(x, y, coef)
    return perte[0] / 100


'''on oublie etape 8/fig 12'''


def table_3(l_aube, pitch_dia, p_out, m_dot):
    A_an = np.pi * ((pitch_dia / 2 + l_aube / 2) ** 2 - (pitch_dia / 2 - l_aube / 2) ** 2)
    l_aube = l_aube / 25.4
    v_in = H2O.v(x=1, p=p_out)
    V_dot = m_dot * 2.204 * v_in * 16.0185
    A_an = A_an * 1.076e-5
    V_an = V_dot / A_an
    x = np.array([35, 38, 43, 52])
    y = np.array(
        [128, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
    valeurs = np.array([[47.92, 42.85, 51.63, 75.69],
                        [42.80, 37.8, 45.5, 65.64],
                        [37.98, 32.63, 38.38, 55.68],
                        [33.63, 28.4, 32.65, 48.05],
                        [26.75, 21.6, 23.8, 36],
                        [21.63, 16.6, 17.43, 27.8],
                        [17.7, 13.13, 12.91, 21.45],
                        [14.55, 10.55, 9.8, 16.30],
                        [12.05, 9, 7.97, 12.5],
                        [10, 8.13, 7.07, 9.6],
                        [8.82, 7.89, 7.02, 8.55],
                        [8.85, 8.4, 7.72, 8.47],
                        [9.63, 9.57, 9, 9.05],
                        [11.08, 11.4, 11, 10.38],
                        [15.22, 16.25, 16.25, 14.45],
                        [20.34, 21.97, 21.97, 19.56],
                        [25.95, 28, 27.92, 25.07],
                        [31.8, 34, 34, 30.82],
                        [37.45, 39.8, 39.92, 36.4],
                        [42.8, 45.18, 45.25, 41.78],
                        [47.45, 49.9, 49.85, 46.5]
                        ]).T
    f = sp.interpolate.RectBivariateSpline(x, y, valeurs)

    return f(l_aube, V_an)[0] * 2.326  # kj/kg


def turbine_3600_HP(mdot_design, mdot_in, gv_stage, h_in, p_in, p_out, pitch_dia):

    TFR = mdot_in / mdot_design

    v_in = H2O.v(h=h_in, p=p_in) * 16.0185  # conversion
    mdot_in_lb = mdot_in * 7936.64  # conversion
    mdot_design = mdot_design * 7936.64
    V_dot_design = mdot_design * v_in
    pitch_dia = pitch_dia / 25.4
    # conversion mm a in
    eta = .87
    if gv_stage == 1:
        V_dot = mdot_in_lb * v_in
        cr_vol = 1005200 / (V_dot * 100)  # *1 parrallel flow section at beggining of expansion
        eta = eta * (1 - cr_vol)
        cr_gvs = fig_7(pitch_dia)
        eta = eta * (1 + cr_gvs)
        eta = eta * (1 + fig_6(p_in, p_out, V_dot_design))
        eta = eta * (1 + fig_8(TFR, pitch_dia))
        perte_9 = fig_9(p_in, p_out, TFR)
        eta = eta * (1 + perte_9)

    if gv_stage == 2:
        eta = .84

    return eta


def turbine_3600_int(h_in, p_in, p_out, m_dot):
    r = p_in / p_out
    A = 90.799 + 0.7474 * (np.log(r - 0.3)) - 0.5454 / (np.log(r - 0.3))
    B = -505000 + 77568 * (np.log(r + 0.8)) - 1262500 / (np.log(r + 0.8))
    v_in = H2O.v(h=h_in, p=p_in) * 16.0185
    V_dot = m_dot * 7936.64 * v_in * 16.0185
    eta = A + B / V_dot

    return eta / 100


def turbine_3600_reheat(h_in, p_in, mdot_in):
    eta = 0.9193
    v_in = H2O.v(h=h_in, p=p_in) * 16.0185  # conversion
    mdot_in_lb = mdot_in * 7936.64  # conversion
    V_dot = mdot_in_lb * v_in

    cr_vol = 1270000 / (V_dot * 100)  # *1 parrallel flow section at beggining of expansion
    eta = eta * (1 - cr_vol)
    eta = eta * (1 + fig_14(p_in, h_in))

    return eta
