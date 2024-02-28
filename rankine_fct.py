import pyromat as pm
import numpy as np
import matplotlib.pyplot as plt

import math_util
import turbine

# pm.config['warning_verbose'] = False
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


def travail(mdot, data):

    # data turbine HP
    m_dot_design = data['turbine'][0]
    pitch_dia = data['turbine'][1]
    gv_stage = data['turbine'][2]
    sout_hp = data['turbine'][3]
    t_hp_in = data['turbine'][4]
    p_hp_in = data['turbine'][5]
    p_hp_out = data['turbine'][6]
    h_hp_in = H2O.h(T=t_hp_in, p=p_hp_in)
    s_hp_in = H2O.s(T=t_hp_in, p=p_hp_in)

    # data turbine LP
    l_aube_last = data['turbine'][7]
    pitch_dia_last = data['turbine'][8]
    p_lp_out = data['turbine'][9]

    # TODO verifier que lentree est en int
    n_etage_int = int(data['turbine'][11])
    beta = np.zeros((n_etage_int + 1))
    P_out = np.zeros((n_etage_int + 1))
    resurchauffe = np.zeros((n_etage_int))

    beta[0] = sout_hp
    P_out[0] = p_hp_out

    for i in range(0, n_etage_int):
        P_out[i+1] = data['int'][3*i]
        beta[i+1] = data['int'][3*i + 1]
        resurchauffe[i] = data['int'][3*i + 2]

    Wdot_int = np.zeros(np.size(mdot))
    h_out = np.zeros((n_etage_int + 2, np.size(mdot)))
    q_resurchauffe = np.zeros((n_etage_int, np.size(mdot)))

    print("Calcul...")
    rend_hp = turbine.turbine_3600_HP(m_dot_design, mdot, gv_stage, h_hp_in, p_hp_in, P_out[0], pitch_dia)
    # TODO verif graphique rendement pour chaque etage vs. graph h_out
    # plt.plot(mdot, rend_hp)
    # plt.title('rend HP')
    # plt.show()

    w_t_hp, h_out[0] = travail_turbine(h_hp_in, s_hp_in, P_out[0], rend_hp)
    w_t_hp = w_t_hp*mdot

    """attention a lindex pour les etages de turbine"""
    print('-')
    for i in range(0, n_etage_int):
        print(f'Etage {i+1}')
        frac = math_util.frac_mdot_turbine(beta, i)
        debit_in = frac*mdot

        if resurchauffe[i] == 0:
            h_in = h_out[i]
            P_in = P_out[i]
            rend = turbine.turbine_3600_int(h_in, P_in, P_out[i+1], debit_in)

            # plt.plot(debit_in, rend)
            # plt.title(f'etage {i}')
            # plt.show()

        else:
            h_res_in = h_out[i]*0.95  # pertes de chaleur
            P_in = P_out[i]*0.95
            h_in = H2O.h(T=resurchauffe[i], p=P_in)
            q_resurchauffe[i] = debit_in*(h_in - h_res_in)
            rend = turbine.turbine_3600_reheat(h_in, P_in, debit_in)

            # plt.plot(debit_in, rend)
            # plt.title(f'etage {i}')
            # plt.show()

        s_in = H2O.s(p=P_in, h=h_in)

        w_int, h_out[i+1] = travail_turbine(h_in,s_in, P_out[i+1], rend)
        Wdot_int += w_int*debit_in
    print('--')
    frac_lp = math_util.frac_mdot_turbine(beta, n_etage_int)
    debit_lp = frac_lp*mdot
    rend_lp = turbine.turbine_3600_reheat(h_out[-2], P_out[-1], debit_lp)

    # plt.plot(debit_lp, rend_lp)
    # plt.title('rend lp')
    # plt.show()

    s_lp = H2O.s(h=h_out[-2], p=P_out[-1])
    w_t_lp, h_out[-1] = travail_turbine(h_out[-2], s_lp, p_lp_out, rend_lp)

    pertes_lp = turbine.table_3(l_aube_last, pitch_dia_last, p_lp_out, debit_lp)

    w_t_lp = (w_t_lp - pertes_lp)*debit_lp

    # plt.plot(debit_lp, (w_t_lp-pertes_lp)/w_t_lp)
    # plt.title('w_t / pertes')
    # plt.show()

    p_cond = p_lp_out
    n_sout = len(beta)
    h_sout_out = H2O.h(x=0, p=p_lp_out)

    # verifier que la temp de rejet de leau est plus basse que la temp dentree du sout
    t_sout_out = H2O.T(h=h_sout_out, p=p_cond)

    mdot_riviere = data['riviere'][0]
    t_riviere_in = data['riviere'][1]
    t_riviere_out = data['riviere'][2]

    perte_vap = data['centrale'][0]
    rend_p = data['centrale'][1]  # rendement isentropique des pompes
    rend_alt = data['centrale'][2]  # rendement de l'alternateur
    rend_chaud = data['centrale'][3]  # rendement de la chaudiere

    if t_riviere_out > t_sout_out:
        t_riviere_out = t_sout_out

    p_amb = 101.25
    h_riviere_in = H2O.h(T=t_riviere_in, p=p_amb)
    h_riviere_out = H2O.h(T=t_riviere_out, p=p_amb)

    frac_sout_out = 0
    for i in range(0, n_sout):
        frac_sout_out += math_util.frac_mdot_sout(beta, i)

    rechauffe_in = frac_sout_out*h_sout_out
    turbine_out = frac_lp*h_out[-1]
    q_out_riviere = mdot_riviere*(h_riviere_out - h_riviere_in)
    h_cond_out = rechauffe_in + turbine_out - q_out_riviere/mdot

    for i, h in enumerate(h_cond_out):
        if h <= 0:
            h_cond_out[i] = 1

    titre_cond_out = np.array(H2O.x(h=h_cond_out, p=p_cond))

    # verfification de letat de la vapeur
    index_mdot = len(mdot) - 1
    for i, x in enumerate(titre_cond_out):

        if x == -1:
            h_cond_out[i] = H2O.h(x=0, p=p_cond)

            titre_cond_out[i] = 0
        if x == 1.:
            if i <= index_mdot:
                # TODO message d'erreur pour non condensation
                index_mdot = i  # quand la vapeur n'est pas condense on considere que le mdot est hors requis
            else:
                pass
        else:
            pass

    alpha = np.copy(titre_cond_out)
    h_p_in = (1-perte_vap*alpha)*H2O.h(x=0, p=p_cond) + H2O.v(x=0, p=p_cond)*(p_amb - p_cond) + alpha*perte_vap*H2O.h(T=t_riviere_in, p = p_amb)
    h_rech_in = h_p_in + H2O.v(h=h_p_in, p=p_amb)*(P_out[0] - p_amb)

    eau_alim = alpha*perte_vap*mdot

    #  chaleur qui sort du soutirage
    h_rech_out = 0
    for i in range(0, n_sout):
        frac = math_util.frac_mdot_sout(beta, i)
        h_rech_out += frac*h_out[i]

    # eau qui part vers le condenseur

    h_rech_out += h_rech_in - frac_sout_out*h_sout_out
    print(H2O.x(h=h_rech_out, p=p_hp_out))
    # TODO verifier le titre de leau
    h_chaud_in = h_rech_out + H2O.v(h=h_rech_out, p=p_hp_out)*(p_hp_in - p_hp_out)
    ############################################################################

    w_p_lp = travail_pompe(p_lp_out, p_hp_out)*mdot/rend_p

    w_p_hp = travail_pompe(p_hp_out, p_hp_in)*mdot/rend_p

    puissance_mec = w_t_hp + Wdot_int + w_t_lp - w_p_lp - w_p_hp

    puissance_elec = puissance_mec*rend_alt

    Q_in = (mdot*(h_hp_in - h_chaud_in) + sum(q_resurchauffe))/rend_chaud  # juste condenseur

    rendement = puissance_elec / Q_in

    print('----')

    return puissance_elec, rendement, eau_alim
