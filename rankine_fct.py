import pyromat as pm
import numpy as np
import turbine

pm.config['unit_pressure'] = 'kPa'
pm.config['unit_energy'] = 'kJ'
H2O = pm.get("mp.H2O")


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
    h_8 = alpha*h_3 + r*(1-alpha)*h_7 + (1-r)*(1-alpha)*h_e

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
    Q_in = m_dot*((h_2-h_1) + (1-alpha)*(h_4-h_3))/rend_chaud

    return Q_in


def travail(m_dot, data):

    ret_cond = data['valeur'][0]
    rend_p = data['valeur'][1]  # rendement isentropique des pompes
    rend_alt = data['valeur'][2]  # rendement de l'alternateur
    rend_chaud = data['valeur'][3]  # rendement de la chaudiere
    alpha = data['valeur'][4]  # fraction de soutirage
    t_2 = data['valeur'][5]
    p_2 = data['valeur'][6]
    p_3 = data['valeur'][7]
    p_5 = data['valeur'][8]

    if isinstance(m_dot, float) or isinstance(m_dot, int):
        m_dot = np.array(m_dot)

    p_1 = p_2

    h_2 = H2O.h(T=t_2, p=p_2)
    s_2 = H2O.s(T=t_2, p=p_2)

    m_dot_design = data['turbine'][0]
    pitch_dia = data['turbine'][1]
    gv_stage = data['turbine'][2]

    rend_hp = turbine.turbine_3600_HP(m_dot_design, m_dot, gv_stage, h_2, p_2, p_3, pitch_dia)
    print('-')
    w_t_hp, h_3 = travail_turbine(h_2, s_2, p_3, rend_hp)
    w_p_lp = travail_pompe(p_5, p_3)
    h_6 = H2O.h(p=p_5, x=0)
    h_7 = h_6 + w_p_lp
    h_8 = h_melangeur(alpha, ret_cond, h_3, p_5, h_7)

    p_4 = p_3
    t_4 = t_2
    h_4 = H2O.h(p=p_4, T=t_4)
    s_4 = H2O.s(p=p_4, T=t_4)

    rend_lp = turbine.turbine_3600_reheat(h_4, p_4, (1-alpha)*m_dot)
    print('--')
    w_t_lp, h_5 = travail_turbine(h_4, s_4, p_5, rend_lp)

    l_aube_last = data['turbine'][4]
    pitch_dia_last = data['turbine'][5]
    pertes_lp = turbine.table_3(l_aube_last, pitch_dia_last, p_5, (1-alpha)*m_dot)
    print('---')
    w_p_hp = travail_pompe(p_3, p_1)

    travail_mec = w_t_hp + (1-alpha)*(w_t_lp-pertes_lp) - (ret_cond*(1-alpha)*w_p_lp + w_p_hp)/rend_p
    puissance_mec = m_dot*travail_mec
    puissance_elec = puissance_mec*rend_alt

    h_4_prime = H2O.h(p=p_4, T=t_2)
    Q_in = chaleur_chaud(h_8, alpha, m_dot, w_p_hp, h_2, h_4_prime, h_3, rend_chaud)

    rendement = puissance_elec / Q_in
    print('----')
    return puissance_elec, rendement
