import pyromat as pm
import numpy as np
import math
# import matplotlib.pyplot as plt
import pyfluids as pf

import math_util
import turbine
import pertes
import warnings

pm.config['warning_verbose'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
pm.config['unit_pressure'] = 'kPa'
pm.config['unit_energy'] = 'kJ'
H2O = pm.get("mp.H2O")


class SoutirageInvalide(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class Resurchauffe(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class VapeurChaudiere(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class VapeurCond(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class Etageint(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class Erreur(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


def travail_turbine(h_in, s_in, p_out, rend):
    """


    Parameters
    ----------
    h_in :
        enthalpie a l'entree
    s_in :
        entropie a l'entree
    p_out :
        pression a la sortie
    rend :
        rendement isentropique

    Returns
    -------
    w_t : TYPE
        travail sortant [kJ/kg]
    h_out : TYPE
        enthalpie de la vapeur sortante [kJ/kg]

    """

    h_out_s = H2O.h(p=p_out, s=s_in)
    h_out = h_in - rend * (h_in - h_out_s)
    w_t = (h_in - h_out)

    return w_t, h_out


def travail_pompe(p_in, p_out):
    """


    Parameters
    ----------
    p_in :
        pression a l'entree
    p_out :
        pression a la sortie

    Returns
    -------
    w_p : TYPE
        DESCRIPTION.

    """
    v = H2O.v(p=p_in, x=0)  # volume massique du liquide sature
    w_p = v * (p_out - p_in)

    return w_p


def h_melangeur(alpha, r, h_3, p_m, h_7):
    """


    Parameters
    ----------
    Pour eviter la confusion, les indices sont les indices globaux du cycle
    les enthalpies sont en [kJ/kg]

    alpha :
        fraction de soutirage
    r :
        fraction de retour de condensat
    h_3 :
        Enthalpie de la vapeur sortante de la turbine HP
    p_m :
        Pression dans le melangeur
    h_7 :
        enthalpie de la vapeur sortante de la pompe LP

    Returns
    -------
    h_8 :
        enthalpie a la sortie du melangeur

    """
    h_e = H2O.h(p=p_m, T=293)
    h_8 = alpha * h_3 + r * (1 - alpha) * h_7 + (1 - r) * (1 - alpha) * h_e

    return h_8


# calcul de la chaleur forunie à la vapeur
def chaleur_chaud(h_p_i, alpha, m_dot, w_p_hp, h_2, h_4, h_3, rend_chaud):
    """


    Parameters
    ----------
    h_p_i :
        Enthalpie a l'entree de la pompe HP
    alpha :
        fraction de soutirage
    m_dot :
        debit massique
    w_p_hp :
        travail accomplie par la pompe HP
    h_2 :
        enthalpie en sortie de la chaudiere
    h_4 :
        enthalpie en sortie de la resurchauffe
    h_3 :
        enthalpie en entree de la resurchauffe
    rend_chaud :


    Returns
    -------
    Q_in : kW
        chaleur totale fournie par le carburant

    """
    h_1 = h_p_i + w_p_hp
    Q_in = m_dot * ((h_2 - h_1) + (1 - alpha) * (h_4 - h_3)) / rend_chaud

    return Q_in


def d_v_t(data, BD):
    mdot = data['Turbines'][0]

    tu = BD['Choixtu'][0]
    index_tu = BD.index[BD['Métal'] == tu].tolist()[0]
    Sy = BD['Sy [MPa]'][index_tu]

    FS = data['Centrale'][3]

    pitch_dia = data['Turbines'][1]
    gv_stage = data['Turbines'][2]
    sout_hp = data['Turbines'][3]
    t_hp_in = data['Turbines'][4]
    p_hp_in = data['Turbines'][5]
    p_hp_out = data['Turbines'][6]
    h_hp_in = H2O.h(T=t_hp_in, p=p_hp_in)
    s_hp_in = H2O.s(T=t_hp_in, p=p_hp_in)

    n_etage_int = int(data['Turbines'][11])
    beta = np.zeros((n_etage_int + 1))
    P_out = np.zeros((n_etage_int + 1))
    resurchauffe = np.zeros((n_etage_int + 1))
    resurchauffe[-1] = data['Turbines'][10]

    beta[0] = sout_hp
    P_out[0] = p_hp_out

    for i in range(0, n_etage_int):
        P_out[i + 1] = data['Étages intermédiaires'][3 * i]
        beta[i + 1] = data['Étages intermédiaires'][3 * i + 1]
        resurchauffe[i] = data['Étages intermédiaires'][3 * i + 2]

        if math.isnan(P_out[i + 1]) or math.isnan(beta[i + 1]) or math.isnan(resurchauffe[i]):
            raise Erreur(
                "Une ou plusieurs valeur(s) manquante(s) pour les étages intermédiaires"
            )

    h_out = np.zeros(n_etage_int + 1)

    rend_hp = turbine.turbine_3600_HP(mdot, mdot, gv_stage, h_hp_in, p_hp_in, P_out[0], pitch_dia)

    w_t_hp, h_out[0] = travail_turbine(h_hp_in, s_hp_in, P_out[0], rend_hp)

    dia = np.zeros(n_etage_int + 1)
    v = np.zeros(n_etage_int + 1)
    t = np.zeros(n_etage_int + 1)

    if resurchauffe[0] != 0:
        dia[0], v[0] = pertes.diametre(mdot, p_hp_out, h_out[0])
        t[0] = pertes.epaisseur(p_hp_out, dia[0], Sy, FS)

    for i in range(0, n_etage_int):
        frac = math_util.frac_mdot_turbine(beta, i)
        debit_in = frac * mdot

        if resurchauffe[i] == 0:
            h_in = h_out[i]
            P_in = P_out[i]
            rend = turbine.turbine_3600_int(h_in, P_in, P_out[i + 1], debit_in)
        else:
            dQ, dP = pertes.dQdP(h_out[i], h_out[i], mdot, P_out[i], data, BD, dia[i], v[i], t[i])
            P_in = P_out[i] - dP
            h_in = H2O.h(T=resurchauffe[i], p=P_in)
            rend = turbine.turbine_3600_reheat(h_in, P_in, debit_in)

        s_in = H2O.s(p=P_in, h=h_in)

        w_int, h_out[i + 1] = travail_turbine(h_in, s_in, P_out[i + 1], rend)
        if resurchauffe[i + 1] != 0:
            dia[i + 1], v[i + 1] = pertes.diametre(mdot, p_hp_out, h_out[i + 1])
            t[i + 1] = pertes.epaisseur(p_hp_out, dia[i + 1], Sy, FS)

    if resurchauffe[-1] != 0:
        dia[-1], v[-1] = pertes.diametre(mdot, p_hp_out, h_out[-2])
        t[-1] = pertes.epaisseur(p_hp_out, dia[-1], Sy, FS)

    return dia, v, t, h_out


def travail(mdot, data, BD):
    # data turbine HP
    m_dot_design = data['Turbines'][0]
    pitch_dia = data['Turbines'][1]
    gv_stage = data['Turbines'][2]
    sout_hp = data['Turbines'][3]
    t_hp_in = data['Turbines'][4]
    p_hp_in = data['Turbines'][5]
    p_hp_out = data['Turbines'][6]
    h_hp_in = H2O.h(T=t_hp_in, p=p_hp_in)
    s_hp_in = H2O.s(T=t_hp_in, p=p_hp_in)

    # data turbine LP
    l_aube_last = data['Turbines'][7]
    pitch_dia_last = data['Turbines'][8]
    p_lp_out = data['Turbines'][9]

    n_etage_int = float(data['Turbines'][11])

    if not n_etage_int.is_integer() or n_etage_int < 0 or n_etage_int > 10:
        raise Etageint(
            "Le nombre d'étages intermédiaires doit être un nombre entier positif plus petit que 10"
        )
    n_etage_int = int(n_etage_int)
    beta = np.zeros((n_etage_int + 1))
    P_out = np.zeros((n_etage_int + 1))
    resurchauffe = np.zeros((n_etage_int + 1))
    resurchauffe[-1] = data['Turbines'][10]  # resurchauffe arrivant a la turbine LP
    beta[0] = sout_hp
    P_out[0] = p_hp_out

    for i in range(0, n_etage_int):
        P_out[i + 1] = data['Étages intermédiaires'][3 * i]
        beta[i + 1] = data['Étages intermédiaires'][3 * i + 1]
        resurchauffe[i] = data['Étages intermédiaires'][3 * i + 2]

    #
    for i, b in enumerate(beta):
        if b < 0 or b >= 1:
            raise SoutirageInvalide(f"Une ou plusieurs valeurs de soutirage ne sont pas valides, "
                                    f"le premier est beta{i + 1} = {b} ")

    Wdot_int = np.zeros(np.size(mdot))
    h_out = np.zeros((n_etage_int + 2, np.size(mdot)))
    q_resurchauffe = np.zeros((n_etage_int + 1, np.size(mdot)))
    dQ = np.zeros((n_etage_int + 1, np.size(mdot)))
    dP = np.zeros((n_etage_int + 1, np.size(mdot)))
    rendement = np.zeros((3, np.size(mdot)))
    point_TS = data['Centrale'][18] - 1
    t_graph = np.zeros(10 + 2*(n_etage_int))
    s_graph = np.zeros(10 + 2*(n_etage_int))
    print("Calcul...")

    ############################################################################
    # travail turbines
    ############################################################################

    print('Dimensionnement de la tuyauterie...')
    dia, v, t, h_prop = d_v_t(data, BD)
    t_graph[0] = H2O.T(h=h_hp_in, p=p_hp_in)
    s_graph[0] = H2O.s(h=h_hp_in, p=p_hp_in)

    t_graph[-1] = H2O.T(h=h_hp_in, p=p_hp_in)
    s_graph[-1] = H2O.s(h=h_hp_in, p=p_hp_in)
    rendement[0] = turbine.turbine_3600_HP(m_dot_design, mdot, gv_stage, h_hp_in, p_hp_in, P_out[0], pitch_dia)
    w_t_hp, h_out[0] = travail_turbine(h_hp_in, s_hp_in, P_out[0], rendement[0])
    w_t_hp = w_t_hp * mdot
    t_graph[1] = H2O.T(h=h_out[0][point_TS], p=P_out[0])
    s_graph[1] = H2O.s(h=h_out[0][point_TS], p=P_out[0])

    """attention a lindex pour les etages de turbine"""
    print('Travail des turbines')
    for i in range(0, n_etage_int):
        print(f'Etage {i + 1}')
        frac = math_util.frac_mdot_turbine(beta, i)
        debit_in = frac * mdot

        if resurchauffe[i] == 0:
            h_in = h_out[i]
            P_in = np.full(np.size(mdot), P_out[i])
            rend = turbine.turbine_3600_int(h_in, P_in, P_out[i + 1], debit_in)

        else:
            dQ[i], dP[i] = pertes.dQdP(h_out[i], h_prop[i], debit_in, P_out[i], data, BD, dia[i], v[i], t[i])
            P_in = P_out[i] - dP[i]
            h_res_in = h_out[i] + dQ[i] / debit_in
            T_in = H2O.T(h=h_res_in, p=P_in)
            for j, T in enumerate(T_in):
                if resurchauffe[i] < T:
                    T_in[j] = T
                    raise Resurchauffe(
                        f"La température de resurchauffe pour l'étage intermédiaire {i+1} est plus basse "
                        f"que la température de sortie")
                else:
                    T_in[j] = resurchauffe[i]

            h_in = H2O.h(T=T_in, p=P_in)
            q_resurchauffe[i] = debit_in * (h_in - h_res_in)
            rend = turbine.turbine_3600_reheat(h_in, P_in, debit_in)

        s_in = H2O.s(p=P_in, h=h_in)

        w_int, h_out[i + 1] = travail_turbine(h_in, s_in, P_out[i + 1], rend)
        Wdot_int += w_int * debit_in
        t_graph[2 + 2*i] = H2O.T(h=h_in[point_TS], p=P_in[point_TS])
        s_graph[2 + 2*i] = s_in[point_TS]

        t_graph[3 + 2*i] = H2O.T(h=h_out[i+1][point_TS], p=P_out[i+1])
        s_graph[3 + 2*i] = H2O.s(h=h_out[i+1][point_TS], p=P_out[i+1])

    print('--')
    frac_lp = math_util.frac_mdot_turbine(beta, n_etage_int)
    debit_lp = frac_lp * mdot
    if resurchauffe[-1] == 0:
        h_lp_in = h_out[-2]
        p_lp_in = np.full(np.size(mdot), P_out[-1])
    else:
        dQ[-1], dP[-1] = pertes.dQdP(h_out[-2], h_prop[-1], debit_lp, P_out[-1], data, BD, dia[-1], v[-1], t[-1])
        p_lp_in = P_out[-1] - dP[-1]
        h_res_in = h_out[-2] + dQ[-1] / debit_lp
        T_in = H2O.T(h=h_res_in, p=p_lp_in)
        for i, T in enumerate(T_in):
            if resurchauffe[-1] < T:
                T_in[i] = T
                raise Resurchauffe(
                    f"La température de resurchauffe pour la turbine LP est plus basse que la température de sortie")
            else:
                T_in[i] = resurchauffe[-1]
        h_lp_in = H2O.h(T=T_in, p=p_lp_in)
        q_resurchauffe[-1] = debit_lp * (h_lp_in - h_res_in)
    rend_lp = turbine.turbine_3600_reheat(h_lp_in, p_lp_in, debit_lp)

    s_lp = H2O.s(h=h_lp_in, p=p_lp_in)
    w_t_lp, h_out[-1] = travail_turbine(h_lp_in, s_lp, p_lp_out, rend_lp)

    pertes_lp = turbine.table_3(l_aube_last, pitch_dia_last, p_lp_out, debit_lp)

    w_t_lp = (w_t_lp - pertes_lp) * debit_lp
    h_out_s = H2O.h(p=p_lp_out, s=s_lp)
    rendement[1] = w_t_lp / ((h_lp_in - h_out_s) * debit_lp)

    t_graph[-8] = H2O.T(h=h_lp_in[point_TS], p=p_lp_in[point_TS])
    s_graph[-8] = s_lp[point_TS]
    t_graph[-7] = H2O.T(h=h_out[-1][point_TS], p=p_lp_out)
    s_graph[-7] = H2O.s(h=h_out[-1][point_TS], p=p_lp_out)

    ############################################################################
    # Echanges de chaleur
    ############################################################################
    print('Échanges de chaleur...')
    p_cond = p_lp_out
    n_sout = len(beta)
    h_sout_out = H2O.h(x=0, p=P_out[-1])

    t_sout_out = H2O.T(h=h_sout_out, p=p_cond)

    cap_evap = data["Centrale"][21]
    V_ballon = data['Centrale'][22]

    mdot_riviere = data['Centrale'][9]
    t_riviere_in = data['Centrale'][10]
    t_riviere_out = data['Centrale'][11]

    perte_vap = data['Centrale'][0]
    rend_p = data['Centrale'][1]  # rendement isentropique des pompes
    rend_alt = data['Centrale'][2]  # rendement de l'alternateur

    carburant = BD['Choixc'][0]
    index_carb = BD.index[BD['Carburant'] == carburant].tolist()[0]
    rend_chaud = BD['Rendement'][index_carb]
    GCV = BD['GCV [kJ/kg]'][index_carb]
    kgCO2 = BD['kgCO2/kg'][index_carb]
    # rend_chaud = data['centrale'][3]  # rendement de la chaudiere

    if t_riviere_out > t_sout_out:  # verifier que la temp de rejet de leau est plus basse que la temp dentree du sout
        t_riviere_out = t_sout_out

    p_amb = 101.25
    h_riviere_in = H2O.h(T=t_riviere_in, p=p_amb)
    h_riviere_out = H2O.h(T=t_riviere_out, p=p_amb)

    frac_sout_out = 0
    for i in range(0, n_sout):
        frac_sout_out += math_util.frac_mdot_sout(beta, i)

    rechauffe_out = frac_sout_out * h_sout_out
    turbine_out = frac_lp * h_out[-1]
    q_out_riviere = mdot_riviere * (h_riviere_out - h_riviere_in)
    h_cond_out = rechauffe_out + turbine_out - q_out_riviere / mdot

    t_graph[-6] = H2O.T(h=h_cond_out[point_TS], p=p_lp_out)
    s_graph[-6] = H2O.s(h=h_cond_out[point_TS], p=p_lp_out)

    for i, h in enumerate(h_cond_out):
        if h <= 0:
            h_cond_out[i] = 1

    titre_cond_out = np.array(H2O.x(h=h_cond_out, p=p_cond))
    # verfification de letat de la vapeur
    index_titre_cond = len(mdot) - 1
    vapeur_cond = False
    T_riv = np.full(np.size(mdot), t_riviere_out)

    for i, x in enumerate(titre_cond_out):

        if x == -1:
            h_cond_out[i] = H2O.h(x=0, p=p_cond)
            h_riviere_out = mdot[i]/mdot_riviere*(rechauffe_out + turbine_out[i] - h_cond_out[i]) + h_riviere_in
            T_riv[i] = H2O.T(h=h_riviere_out, p=p_amb)
            titre_cond_out[i] = 0
        if x == 1:
            if i <= index_titre_cond:
                index_titre_cond = i  # quand la vapeur n'est pas condense on considere que le mdot est hors requis
                vapeur_cond = True
            else:
                pass
        else:
            pass

    if index_titre_cond == 0:
        raise VapeurCond(
            "L'eau sortante du condenseur est entièrement en phase vapeur pour tous les débits massiques"
        )

    alpha = np.copy(titre_cond_out)
    h_p_in = (1 - perte_vap * alpha) * (H2O.h(x=0, p=p_cond) + H2O.v(x=0, p=p_cond) * (p_amb - p_cond)) \
        + alpha * perte_vap * h_riviere_in

    t_graph[-5] = H2O.T(h=h_p_in[point_TS], p=p_amb)
    s_graph[-5] = H2O.s(h=h_p_in[point_TS], p=p_amb)

    h_rech_in = h_p_in + H2O.v(h=h_p_in, p=p_amb) * (P_out[0] - p_amb)

    t_graph[-4] = H2O.T(h=h_rech_in[point_TS], p=P_out[0])
    s_graph[-4] = H2O.s(h=h_rech_in[point_TS], p=P_out[0])

    h_eau = h_rech_in[int(np.size(mdot)/2)]
    v_eau = 1  # m/s
    rho_eau = 1 / H2O.v(p=p_hp_out, h=h_eau)
    dia_eau = np.sqrt(mdot[i] * 4 / (rho_eau * np.pi * v_eau))
    eau = pf.Fluid(pf.FluidsList.Water).with_state(
        pf.Input.pressure(p_hp_out * 1e3), pf.Input.enthalpy(h_eau * 1e3)  # en Pa et J
    )
    mu = eau.dynamic_viscosity
    Re = 4 * mdot[i] / (np.pi * dia_eau * mu)
    tu = BD['Choixtu'][0]
    index_tu = BD.index[BD['Métal'] == tu].tolist()[0]
    rugosite = BD['Rugosité'][index_tu]
    L_eau = data['Centrale'][17] * (n_etage_int + 1)
    pcharge = pertes.p_charge(v_eau, Re, rugosite, dia_eau, L_eau, h_eau, p_hp_out)

    ############################################################################
    ############################################################################
    eau_alim = alpha * perte_vap * mdot

    #  chaleur qui sort du soutirage
    h_rech_out = 0
    for i in range(0, n_sout):
        frac = math_util.frac_mdot_sout(beta, i)
        h_rech_out += frac * h_out[i]

    # eau qui part vers le condenseur

    h_rech_out += h_rech_in - frac_sout_out * h_sout_out
    titre_chaud_in = H2O.x(h=h_rech_out, p=p_hp_out)

    t_graph[-3] = H2O.T(h=h_rech_out[point_TS], p=P_out[0])
    s_graph[-3] = H2O.s(h=h_rech_out[point_TS], p=P_out[0])

    index_titre_chaud = 0
    nb_valide = 0
    vapeur_chaud = False
    for k, x in enumerate(titre_chaud_in):
        if x < 5e-3:
            index_titre_chaud = k
            break

    for k, x in enumerate(titre_chaud_in):
        if x < 5e-3:  # tolerance de contenu de vapeur
            nb_valide += 1
        else:
            vapeur_chaud = True
    if nb_valide == 0:
        raise VapeurChaudiere(
            "L'eau entrante dans la chaudière contient trop de vapeur pour tous les débits massiques considérés"
        )
    else:
        int_nonvap = [index_titre_chaud, index_titre_chaud + nb_valide - 1]  # premier inclu, dernier exclu

    h_chaud_in = h_rech_out + H2O.v(h=h_rech_out, p=p_hp_out) * (p_hp_in - p_hp_out)

    t_graph[-2] = H2O.T(h=h_chaud_in[point_TS], p=p_hp_in)
    s_graph[-2] = H2O.s(h=h_chaud_in[point_TS], p=p_hp_in)

    rho_eau = 1 / H2O.v(h=h_chaud_in, p=p_hp_in)

    t_surdemande = rho_eau * (0.25 * V_ballon) / (3600 * (mdot - cap_evap))
    ############################################################################

    w_p_lp = travail_pompe(p_lp_out, p_hp_out) * mdot / rend_p

    w_p_hp = travail_pompe(p_hp_out - pcharge, p_hp_in) * mdot / rend_p

    puissance_mec = w_t_hp + Wdot_int + w_t_lp - w_p_lp - w_p_hp

    puissance_elec = puissance_mec * rend_alt

    Q_in = (mdot * (h_hp_in - h_chaud_in) + sum(q_resurchauffe)) / rend_chaud
    rendement[2] = puissance_elec / Q_in

    emissions = kgCO2 / GCV / rendement[2] * 3600

    print('----')

    return puissance_elec, rendement, eau_alim, w_p_lp, w_p_hp, t_surdemande, emissions, vapeur_cond, \
        index_titre_cond, vapeur_chaud, int_nonvap, t, dia, T_riv, t_graph, s_graph
