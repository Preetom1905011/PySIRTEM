import numpy as np
from ddeint import ddeint
from copy import deepcopy
from Parameters_pf import *
import csv
import time
import pickle
import matplotlib.pyplot as plt

def ddeModel(Y, t, d, parameters):
    y = {k: Y(t)[v] for k,v in M_states.items()}
    dy = {k: 0 for k,v in M_states.items()}


    try:
        params, inf_rate, phi_s, g_rate, q_days = parameters
        # update length of Quarantine Days
        # d["lambda_q"] = q_days[max(0, min(int(t), len(q_days)-1))]
        params, d["lambda_q"] = paramsUpdateWithQDays(params, t, inf_rate, phi_s, g_rate, q_days, n, m)
    except:
        params, inf_rate, phi_s, g_rate = parameters        
        # update parameter values based on Timepoint
        params = paramsUpdate(params, t, inf_rate, phi_s, g_rate, n, m)
        
    # params = paramsUpdateDaily(params, t, inf_rate, phi_s, g_rate, n, m)
    # print("Days: ", t,  params['beta'], params['phi_s_1'], params['g_beta'])
    
    

    # DDE equations
    # Infected
    Infected = params['r'] * (y['PS'] + y['PA'] + y['IA'] + y['ATN']) + y['IS'] + y['STN']
    # Susceptible: S
    dy['S'] = (y['NTN'] + y['GTN'] + (1 - (params['phi_a_1'] + params['phi_a_2'])) * Y(t - d['gamma'])[M_states['F_FPS']] + y['FTN'] 
                    + (1 - (params['phi_s_1'] + params['phi_s_2'])) * (y['SF'] + y['GS']) + params['True_N_SE'] * Y(t - d['sigma'])[M_states['F_SRE']]
                    - params['beta'] * y['S'] * (Infected / N) - (params['ili'] + params['g_beta'] + params['phi_a_1'] + params['phi_a_2']) * y['S'] )
    # Exposed: E
    dy['E'] = (params['beta'] * y['S'] + params['beta_prime'] * y['FPI']) * (Infected / N) - (params['per_a'] + params['per_s']) * y['E']
    
    # -------------- Asymptomatic Process -------------------
    # Fictitious State before PA: F_PA 
    # --> dF_PA = per_a*E(t) - F_PA(t)
    dy['F_PA'] = params['per_a'] * y['E'] - y['F_PA']
    
    # Pre Asymptomatic: PA 
    # --> dPA/dt = F_PA(t)-F_PA(t-eta)
    dy['PA'] = y['F_PA'] - Y(t - d['eta'])[M_states['F_PA']]

    # Infected Asymptomatic : IA 
    # --> dIA = F_PA(t - eta) + ATN(t) - lambda_a*IA(t)- phi_a_1*IA(t)- phi_a_2*IA(t)
    dy['IA'] = Y(t - d['eta'])[M_states['F_PA']] + y['ATN'] - params['lambda_a'] * y['IA'] - (params['phi_a_1'] + params['phi_a_2']) * y['IA']

    # Fictitious state before AT_1: F_AT_1 
    # --> dF_AT_1 = phi_a_1*IA(t) - F_AT_1(t)
    dy['F_AT_1'] = params['phi_a_1'] * y['IA'] - y['F_AT_1']

    # Asymptomatic Test 1 : AT_1 
    # --> dAT_1 = F_AT_1(t) - F_AT_1(t-tau)
    dy['AT_1'] = y['F_AT_1'] - Y(t - d['tau_1'])[M_states['F_AT_1']]
        
    # Fictitious state before AT_2: F_AT_2 
    # --> dF_AT_2 = phi_a_2 * IA(t) - F_AT_2(t)
    dy['F_AT_2'] = params['phi_a_2'] * y['IA'] - y['F_AT_2']

    # Asymptomatic Test 2 : AT_2
    # --> dAT_2 = F_AT_2(t) - F_AT_2(t-tau)
    dy['AT_2'] = y['F_AT_2'] - Y(t - d['tau_2'])[M_states['F_AT_2']]

    # Fictitious state before QAP: F_QAP
    # --> dF_QAP = True_P_1*F_AT_1(t-tau)+True_P_2*F_AT_2(t-tau)-F_QAP(t);
    dy['F_QAP'] = params['True_P_1'] * Y(t - d['tau_1'])[M_states['F_AT_1']] + params['True_P_2'] * Y(t - d['tau_2'])[M_states['F_AT_2']] - y['F_QAP']
    
    # Quarantined Asymp. Positive: QAP
    # --> dQAP/dt = F_QAP(t) - F_QAP(t-lambda_q)
    dy['QAP'] = y['F_QAP'] - Y(t - d['lambda_q'])[M_states['F_QAP']]

    # Asymptomatic Tested Negative: ATN
    # dATN/dt = (1-True_P_1)*F_AT_1(t-tau)+(1-True_P_2)*F_AT_2(t-tau)-ATN(t)
    dy['ATN'] = (1 - params['True_P_1']) * Y(t - d['tau_1'])[M_states['F_AT_1']] + (1 - params['True_P_2']) * Y(t - d['tau_2'])[M_states['F_AT_2']] - y['ATN']

    # ------------- Systematic Testing Process --------------
    # Fictitious state before PS: F_PS
    # --> dF_PS/dt = per_s*E(t) - F_PS(t)
    dy['F_PS'] = params['per_s'] * y['E'] - y['F_PS']

    # Pre Symptomatic : PS
    # --> dPS/dt = F_PS(t) - F_PS(t-omega)
    dy['PS'] = y['F_PS'] - Y(t - d['omega'])[M_states['F_PS']]

    # Infected Symptomatic : IS
    # --> dIS/dt = F_PS(t-omega) + STN(t) - phi_s_1*IS(t)-phi_s_2*IS(t)-(lambda_s)*IS(t)- kappa_s_1*IS(t)-hos_2*IS(t);
    dy['IS'] = Y(t - d['omega'])[M_states['F_PS']] + y['STN'] - (params['phi_s_1'] + params['phi_s_2'] + params['lambda_s'] + params['kappa_s_1'] + params['hos_2']) * y['IS']    
    # print(t, "==>", t - d['omega'], Y(t - d['omega'])[M_states['F_PS']] + y['STN'], (params['phi_s_1'] + params['phi_s_2'] + params['lambda_s'] + params['kappa_s_1'] + params['hos_2']) * y['IS'], dy['IS'])
    
    # Fictitious state before ST_1: F_ST_1
    # --> dF_ST_1/dt = phi_s_1*IS(t)-F_ST_1(t)

    dy['F_ST_1'] = params['phi_s_1'] * y['IS'] - y['F_ST_1']

    # Symptomatic Test 1 : ST_1
    # --> dST_1/dt = F_ST_1(t) - F_ST_1(t-tau)
    dy['ST_1'] = y['F_ST_1'] - Y(t - d['tau_1'])[M_states['F_ST_1']]

    # Fictitious state before ST_2:  F_ST_2 
    # --> dF_ST_2/dt = phi_s_2*IS(t)-F_ST_2(t)
    dy['F_ST_2'] = params['phi_s_2'] * y['IS'] - y['F_ST_2']

    # Symptomatic Test 2 : ST_2
    # --> dST_2/dt = F_ST_2(t) - F_ST_2(t-tau)
    dy['ST_2'] = y['F_ST_2'] - Y(t - d['tau_2'])[M_states['F_ST_2']]

    # Fictitious state before QSP: F_QSP 
    # --> dST_2/dt =True_P_1*F_ST_1(t-tau)+True_P_2* F_ST_2(t-tau)-F_QSP(t)
    dy['F_QSP'] = params['True_P_1']*Y(t - d['tau_1'])[M_states['F_ST_1']] + params['True_P_2']*Y(t - d['tau_2'])[M_states['F_ST_2']] - y['F_QSP']

    # Quarantined Symptomatic Positive : QSP
    # --> dQSP/dt =F_QSP(t) - F_QSP(t-lambda_q)
    # dQSP/dt = (1-(hos_1+kappa_s_2))*F_QSP(t)-(1-(hos_1+kappa_s_2))*Ylag6(52);
    dy['QSP'] = (1 - (params['hos_1'] + params['kappa_s_2'])) * ( y['F_QSP'] - Y(t - d['lambda_q'])[M_states['F_QSP']])

    # Symptomatic Test Negative : STN
    # --> dSTN/dt =(1-True_P_1)*F_ST_1(t-tau)+(1-True_P_2)*F_ST_2(t-tau)-STN(t)
    dy['STN'] = (1 - params['True_P_1']) * Y(t - d['tau_1'])[M_states['F_ST_1']] + (1 - params['True_P_2']) * Y(t - d['tau_2'])[M_states['F_ST_2']] - y['STN']

    # Fictitious state before H1:  F_H1
    # --> dF_H1/dt = hos_1*F_QSP(t)+(1-True_N_1)*F_ST_4(t-tau)-F_H1(t)
    dy['F_H1'] = params['hos_1'] * y['F_QSP'] + (1 - params['True_N_1']) * Y(t - d['tau_1'])[M_states['F_ST_4']] - y['F_H1']

    # Hospitalized : H1
    # --> dH1/dt = (1- kappa_h_1)*F_H1(t)-(1- kappa_h_1)*F_H1(t-lambda_h_1);
    dy['H_1'] = (1 - params['kappa_h_1']) * (y['F_H1'] - Y(t - d['lambda_H_1'])[M_states['F_H1']])

    # F_ST_4 : Fictitious state before ST_4
    # --> dF_ST_4/dt = (1- kappa_h_1)*F_H1(t-lambda_h_1)-F_ST_4(t);
    dy['F_ST_4'] = (1 - params['kappa_h_1']) * Y(t - d['lambda_H_1'])[M_states['F_H1']] - y['F_ST_4']

    # Symptomatic Test 4 : ST_4
    # --> dST_4/dt = F_ST_4(t)-F_ST_4(t-tau_1)
    dy['ST_4'] = y['F_ST_4'] - Y(t - d['tau_1'])[M_states['F_ST_4']]

    # Fictitious state before H2: F_H2
    # --> dF_H2/dt = hos_2*IS(t)+(1-True_N_1)*F_ST_3(t - tau_1) - F_H2(t)
    dy['F_H2'] = params['hos_2'] * y['IS'] + (1 - params['True_N_1']) * Y(t - d['tau_1'])[M_states['F_ST_3']] - y['F_H2']

    # Hospitalzied 2 : H2
    # --> dH2/dt = (1-kappa_h_2)* F_H2(t)-(1-kappa_h_2)* F_H2(t - lambda_h_2);
    dy['H_2'] = (1 - params['kappa_h_2']) * (y['F_H2'] - Y(t - d['lambda_H_2'])[M_states['F_H2']])

    # Fictitious state before ST_3:  F_ST_3
    # --> dF_ST_3 = (1-kappa_h_2)*F_H2(t - lambda_h_2)-F_ST_3(t)
    dy['F_ST_3'] = (1 - params['kappa_h_2']) * Y(t - d['lambda_H_2'])[M_states['F_H2']] - y['F_ST_3']

    # Symptomatic Test 3 : ST_3
    # --> dST_3= dF_ST_3-F_ST_3(t - tau_1);
    dy['ST_3'] = y['F_ST_3'] - Y(t - d['tau_1'])[M_states['F_ST_3']]

    # # Known Recover : KR#
    # # --> KR(t+1) = QAP(t - lambda_q) + lambda_q*QSP(t) + lambda_q*QAP_1(t);
    # # dKR/dt = UR(t) + F_QAP(t - lambda_q) + (1-(hos_1+kappa_s_2))*F_QSP(t - lambda_q)+F_QAP_1(t - lambda_q)+True_N_1*(F_ST_3(t-tau_1) + F_ST_4(t-tau_1))-KR(t);
    # dy['KR'] = y['UR'] + Y(t - d['lambda_q'])[M_states['F_QAP']] + (1 - (params['hos_1'] + params['kappa_s_2'])) * Y(t - d['lambda_q'])[M_states['F_QSP']] + Y(t - d['lambda_q'])[M_states['F_QAP_1']] + params['True_N_1'] * (Y(t - d['tau_1'])[M_states['F_ST_3']] + Y(t - d['tau_1'])[M_states['F_ST_4']]) - y['KR']

    # ---------- Xin Code -------------
    dy['KR'] = Y(t - d['lambda_q'])[M_states['F_QAP']] + (1 - (params['hos_1'] + params['kappa_s_2'])) * Y(t - d['lambda_q'])[M_states['F_QSP']] + Y(t - d['lambda_q'])[M_states['F_QAP_1']] + params['True_N_1'] * (Y(t - d['tau_1'])[M_states['F_ST_3']] + Y(t - d['tau_1'])[M_states['F_ST_4']]) - y['KR']



    # Dead : D
    # --> dD/dt = kappa_s_1*IS(t) + kappa_s_2*F_QSP(t) + kappa_h_1*F_H1(t) + kappa_h_2*F_H2(t);  
    dy['D'] = params['kappa_s_1'] * y['IS'] + params['kappa_s_2'] * y['F_QSP'] + params['kappa_h_1'] * y['F_H1'] + params['kappa_h_2'] * y['F_H2']


    # -------------- Flu Process --------------
    
    # Susceptible with Flu : SF (flow chart 45)
    # --> dSF/dt = ili*S(t)- phi_s_1*SF(t) - phi_s_2*SF(t)-(1-phi_s_1-phi_s_2)*SF(t);
    dy['SF'] = params['ili'] * y['S'] - params['phi_s_1'] * y['SF'] - params['phi_s_2'] * y['SF'] - (1 - params['phi_s_1'] - params['phi_s_2']) * y['SF']

    # Fictitious state before FT_1: F_FT_1 
    # --> dF_FT_1/dt = phi_s_1*SF(t) - F_FT_1(t)
    dy['F_FT_1'] = params['phi_s_1'] * y['SF'] - y['F_FT_1']

    # Flu like symptoms Test 1 : FT_1
    # --> dFT_1/dt = F_FT_1(t) -F_FT_1(t-tau_1)
    dy['FT_1'] = y['F_FT_1'] - Y(t - d['tau_1'])[M_states['F_FT_1']]

    # Fictitious state before FT_2:  F_FT_2
    # --> dF_FT_2/dt = phi_s_2*SF(t) - F_FT_2(t)
    dy['F_FT_2'] = params['phi_s_2'] * y['SF'] - y['F_FT_2']

    # Flu like symptoms Test 2 : FT_2
    # --> dFT_2/dt = F_FT_2(t) -F_FT_2(t-tau_2)
    dy['FT_2'] = y['F_FT_2'] - Y(t - d['tau_2'])[M_states['F_FT_2']]

    # Fictitious state before QFS:  F_QFS
    # --> dF_QFS/dt = (1-True_N_1)*F_FT_1(t-tau) + (1-True_N_2)*F_FT_2(t-tau)- F_QFS(t)
    dy['F_QFS'] = (1 - params['True_N_1']) * Y(t - d['tau_1'])[M_states['F_FT_1']] + (1 - params['True_N_2']) * Y(t - d['tau_2'])[M_states['F_FT_2']] - y['F_QFS']

    # Quarantined flu like symptoms : QFS
    # --> dQFS/dt =(1-hos_3)*F_QFS(t)-(1-hos_3)*F_QFS(t - lambda_q)
    dy['QFS'] = (1 - params['hos_3']) * (y['F_QFS'] - Y(t - d['lambda_q'])[M_states['F_QFS']])

    # Flu like Symptoms Test Negative : FTN
    # --> dFTN/dt = True_N_1*F_FT_1(t-tau_1) + True_N_2*F_FT_2(t-tau_2)-FTN(t)
    dy['FTN'] = params['True_N_1'] * Y(t - d['tau_1'])[M_states['F_FT_1']] + params['True_N_2'] * Y(t - d['tau_2'])[M_states['F_FT_2']] - y['FTN']

    # Fictitious state before H3:  F_H3
    # --> dF_H3/dt = hos_3*F_QFS(t)-F_H3(t)+(1-True_N_1)*F_FT_3(t-tau)
    dy['F_H3'] = params['hos_3'] * y['F_QFS'] - y['F_H3'] + (1 - params['True_N_1']) * Y(t - d['tau_1'])[M_states['F_FT_3']]

    # Hospitalized 3 : H3
    # --> dH3/dt = F_H3(t) - F_H3(t-lambda_h)
    dy['H_3'] = y['F_H3'] - Y(t - d['lambda_H_3'])[M_states['F_H3']]

    # Fictitious state before FT_3: F_FT_3
    # --> dF_FT_3/dt = F_H3(t-lamba_h3) - F_FT_3(t)
    dy['F_FT_3'] = Y(t - d['lambda_H_3'])[M_states['F_H3']] - y['F_FT_3']

    # Flu like symptoms Test 3 : FT_3
    # --> dFT_3/dt = F_FT_3(t) - F_FT_3(t-tau)
    dy['FT_3'] = y['F_FT_3'] - Y(t - d['tau_1'])[M_states['F_FT_3']]


    # ------------ Non_Infected Process ---------------

    # Fictitious state before NT_1: F_NT_1 
    # --> dF_NT_1/dt = phi_a_1*S(t) - F_NT_1(t)
    dy['F_NT_1'] = params['phi_a_1'] * y['S'] - y['F_NT_1']

    # Non infected Test 1 :  NT_1
    # dNT_1/dt = F_NT_1(t) - F_NT_1(t-tau_1)
    dy['NT_1'] = y['F_NT_1'] - Y(t - d['tau_1'])[M_states['F_NT_1']]

    # Fictitious state before NT_2:  F_NT_2
    # --> dF_NT_2/dt = phi_a_2*S(t) - F_NT_2(t)
    dy['F_NT_2'] = params['phi_a_2'] * y['S'] - y['F_NT_2']

    # Non infected Test 2 :  NT_2
    # --> dNT_2/dt = F_NT_2(t) + F_NT_2(t-tau)
    dy['NT_2'] = y['F_NT_2'] - Y(t - d['tau_2'])[M_states['F_NT_2']]

    # Fictitious state before NTP: F_NTP 
    # --> dF_NTP/dt = (1-True_N_1)*F_NT_1(t-tau) +(1-True_N_2)*F_NT_2(t-tau) - F_NTP(t)
    dy['F_NTP'] = (1 - params['True_N_1']) * Y(t - d['tau_1'])[M_states['F_NT_1']] + (1 - params['True_N_2']) * Y(t - d['tau_2'])[M_states['F_NT_2']] - y['F_NTP']

    # Quarantined Non infected test positvie : NTP
    # --> dNTP/dt = F_NTP(t) + F_NTP(t-lambda_q)
    dy['NTP'] = y['F_NTP'] - Y(t - d['lambda_q'])[M_states['F_NTP']]

    # Non infected Test Negative : NTN 
    # --> dNTN/dt = True_N_1*F_NT_1(t-tau_1) +True_N_2*F_NT_2(t-tau_2) -NTN(t)
    dy['NTN'] = params['True_N_1'] * Y(t - d['tau_1'])[M_states['F_NT_1']] + params['True_N_2'] * Y(t - d['tau_2'])[M_states['F_NT_2']] - y['NTN']

    # General Sick Process: GS
    # --> dGS/dt = g_beta*S(t)-phi_s_1*GS(t)-phi_s_2*GS(t)-(1-phi_s_1-phi_s_2)*GS(t);
    dy['GS'] = params['g_beta'] * y['S'] - params['phi_s_1'] * y['GS'] - params['phi_s_2'] * y['GS'] - (1 - params['phi_s_1'] - params['phi_s_2']) * y['GS']

    # Fictitious before GT_1: F_GT1
    # --> dF_GT1/dt = phi_s_1*GS(t) - F_GT1(t);
    dy['F_GT1'] = params['phi_s_1'] * y['GS'] - y['F_GT1']

    # General Sick Test:  GT_1 (state 41 of flow chart)
    # --> dGT_1/dt= F_GT1(t) - F_GT1(t - tau_1);
    dy['GT_1'] = y['F_GT1'] - Y(t - d['tau_1'])[M_states['F_GT1']]

    # Fictitious before GT_2: F_GT_2 
    # --> dF_GT2/dt = phi_s_2*GS(t) - F_GT2(t);
    dy['F_GT2'] = params['phi_s_2'] * y['GS'] - y['F_GT2']

    # General Sick Test:  GT_2 (state 42 of flow chart)
    # --> dGT_2/dt= F_GT2(t) - F_GT2(t - tau_2);
    dy['GT_2'] = y['F_GT2'] - Y(t - d['tau_2'])[M_states['F_GT2']]

    # Fictitious before QGP: F_QGP
    # --> dF_QGP/dt = (1-True_N_1)*F_GT1(t - tau_1)+(1-True_N_2)*F_GT2(t - tau_2)-F_QGP(t)
    dy['F_QGP'] = (1 - params['True_N_1']) * Y(t - d['tau_1'])[M_states['F_GT1']] + (1 - params['True_N_2']) * Y(t - d['tau_2'])[M_states['F_GT2']] - y['F_QGP']

    # General Sick Test Negative: GTN (state 44 of flow chart)
    # --> dGTN/dt = True_N_1*F_GT1(t - tau_1) + True_N_2*F_GT2(t - tau_2)-GTN(t)
    dy['GTN'] = params['True_N_1'] * Y(t - d['tau_1'])[M_states['F_GT1']] + params['True_N_2'] * Y(t - d['tau_2'])[M_states['F_GT2']] - y['GTN']

    # Quarantined General Sick: QGP (state 43 of flow chart)
    # --> dQGP/dt= F_QGP(t) - F_QGP(t - lambda_q);
    dy['QGP'] = y['F_QGP'] - Y(t - d['lambda_q'])[M_states['F_QGP']]

    # F_FPS : Fictitious state before FPS
    #dy['F_FPS'] = 0

    # ------------ Xin Code ------------------------
    dy['F_FPS'] = y['UR'] + y['ATN_1'] + (1 - params['True_P_SE']) * Y(t - d['sigma'])[M_states['F_STI']] - y['F_FPS']

    # Falsely Presumed Susceptible : FPS
    # --> dY(25)=(1-(phi_a_1+phi_a_2))*Y(65)-(1-(phi_a_1+phi_a_2))*Ylag5(65);
    dy['FPS'] = (1 - (params['phi_a_1'] + params['phi_a_2'])) * (y['F_FPS'] - Y(t - d['gamma'])[M_states['F_FPS']])

    # Fictitious state AT_3: F_AT_3
    # --> dF_AT_3/dt = phi_a_1*F_FPS(t) - F_AT_3(t);
    dy['F_AT_3'] = params['phi_a_1'] * y['F_FPS'] - y['F_AT_3']

    # Asymptomatic Test 3 : AT_3
    # --> dAT_3/dt = F_AT_3(t) - F_AT_3(t-tau)
    dy['AT_3'] = y['F_AT_3'] - Y(t - d['tau_1'])[M_states['F_AT_3']]

    # Fictitious state AT_4: F_AT_4
    # --> dF_AT_4/dt = phi_a_2*F_FPS(t) - F_AT_4(t);
    dy['F_AT_4'] = params['phi_a_2'] * y['F_FPS'] - y['F_AT_4']

    # Asymptomatic Test 4 : AT_4
    # --> dAT_4/dt = F_AT_4(t) - F_AT_4(t-tau);
    dy['AT_4'] = y['F_AT_4'] - Y(t - d['tau_2'])[M_states['F_AT_4']]

    # Fictitious state QAP_1: F_QAP_1
    # --> dF_QAP_1/dt = (1-True_N_1)*F_AT_3(t)+(1-True_N_2)*F_AT_4(t)-F_QAP_1(t); 
    dy['F_QAP_1'] = (1 - params['True_N_1']) * Y(t - d['tau_1'])[M_states['F_AT_3']] + (1 - params['True_N_2']) * Y(t - d['tau_2'])[M_states['F_AT_4']] - y['F_QAP_1']

    # ????????????????????
    # Quarantined Asymptomatic Positve_1 : QAP_1
    # dY(27)=Y(55)-Ylag6(55);
    # --> dQAP_1/dt = F_QAP_1(t) -F_QAP_1(t-lambda_q)
    dy['QAP_1'] = y['F_QAP_1'] - Y(t - d['lambda_q'])[M_states['F_QAP_1']]
    # ????????????????????

    # Asymptomatic Tested Negative_1 : ATN_1
    # --> dATN_1/dt = (True_N_1)*F_AT_3(t - tau_1)+(True_N_2)*F_AT_4(t - tau_2)-ATN_1(t); 
    dy['ATN_1'] = params['True_N_1'] * Y(t - d['tau_1'])[M_states['F_AT_3']] + params['True_N_2'] * Y(t - d['tau_2'])[M_states['F_AT_4']] - y['ATN_1']


    # ----------- Immunity Process -------------

    # Fictitious state before IM: F_IM
    # --> dF_IM/dt = KR(t) - F_IM(t)
    dy['F_IM'] = y['KR'] - y['F_IM']

    # Immunity : IM
    # --> dIM/dt = (1-phi_se) * F_IM(t) - (1-phi_se) * F_IM(t-gamma) + True_P_SE*F_STI(t-sigma)
    dy['IM'] = (1 - params['phi_se']) * ( y['F_IM'] - Y(t - d['gamma'])[M_states['F_IM']]) + params['True_P_SE'] * Y(t - d['sigma'])[M_states['F_STI']]

    # Fictitious state before STI: F_STI
    # --> dF_STI = phi_se*F_IM(t) - F_STI(t)
    dy['F_STI'] = params['phi_se'] * y['F_IM'] - y['F_STI']

    # # Serology Test for Immuned : STI 
    # # --> dSTI/dt = F_STI(t) - F_STI(t-sigma)
    # dy['STI'] = y['F_STI'] - params['True_P_SE'] * Y(t - d['sigma'])[M_states['F_STI']]

    # ---------- Xin Code --------------
    dy['STI'] = y['F_STI'] - Y(t - d['sigma'])[M_states['F_STI']]

    # # Falsely Presumed Immune : FPI
    # # dFPI = F_QGP(t-lambda_q) + F_NTP(t-lambda_q) + (1-hos_3)*F_QFS(t-lambda_q) + (1-phi_se)*F_IM(t-gamma) + (1-True_N_SE)*F_SRE(t-sigma)- phi_se*FPI(t) - beta_prime*FPI(t)*(Infected(t))/N);
    # dy['FPI'] = (Y(t - d['lambda_q'])[M_states['F_QGP']] + Y(t - d['lambda_q'])[M_states['F_NTP']] + (1 - params['hos_3']) * Y(t - d['lambda_q'])[M_states['F_QFS']]
    #                     + (1 - params['phi_se']) * Y(t - d['gamma'])[M_states['F_IM']] + (1 - params['True_N_SE']) * Y(t - d['sigma'])[M_states['F_SRE']] 
    #                     - params['phi_se'] * y['FPI'] - params['beta_prime'] * y['FPI'] * Infected / N) 
    
    # ------------ Xin Code ------------------
    dy['FPI'] = (Y(t - d['lambda_q'])[M_states['F_QGP']] + Y(t - d['lambda_q'])[M_states['F_NTP']] + (1 - params['hos_3']) * Y(t - d['lambda_q'])[M_states['F_QFS']]
                        + (1 - params['phi_se']) * Y(t - d['gamma'])[M_states['F_IM']] + (1 - params['True_N_SE']) * Y(t - d['sigma'])[M_states['F_SRE']] 
                        + params['True_N_1'] * Y(t - d['tau_1'])[M_states['F_FT_3']] - params['phi_se'] * y['FPI'] - params['beta_prime'] * y['FPI'] * Infected / N) 
    


    # Fictitious state before SRE: F_SRE
    # --> dF_SRE/dt = phi_se*FPIt) - F_SRE(t) 
    dy['F_SRE'] = params['phi_se'] * y['FPI'] - y['F_SRE']

    # Serology Test for Immune Expired: SREf
    # --> dSRE/dt = F_SRE(t) - F_SRE(t-sigma) 
    dy['SRE'] = y['F_SRE'] - Y(t - d['sigma'])[M_states['F_SRE']]
        
    # Unknown Recover : UR
    # --> dUR/dt = lambda_a*IA(t) + lambda_s*IS(t) - UR(t);
    dy['UR'] = params['lambda_a'] * y['IA'] + params['lambda_s'] * y['IS'] - y['UR']

    return list(dy.values())

# piecewise dynamics of parameters (User Defined)
def paramsUpdate(params, t, inf_rate, phi_s, g_rate, n, m):
    beta, beta_prime, phi_s_1, g_beta = -np.inf, -np.inf, -np.inf, -np.inf
    params_new = deepcopy(params)
    
    if t <= n:
        beta = inf_rate[0]
        beta_prime = beta * 1.2
        phi_s_1 = phi_s[0]
        g_beta = g_rate[0]
    else:
        for i in range(len(inf_rate) - 1):
            if t > n + m * i and t <= n + m * (i + 1):
                beta = inf_rate[i + 1]
                beta_prime = beta * 1.2
                phi_s_1 = phi_s[i + 1]
                g_beta = g_rate[i + 1]
                break
    
    if beta != -np.inf:
        params_new['beta'] = beta
        params_new['beta_prime'] = beta_prime
        params_new['phi_s_1'] = phi_s_1
        params_new['g_beta'] = g_beta
    else:
        params_new['beta'] = inf_rate[-1]
        params_new['beta_prime'] = params_new['beta'] * 1.2
        params_new['phi_s_1'] = phi_s[-1]
        params_new['g_beta'] = g_rate[-1]

    return params_new


# piecewise dynamics of parameters (User Defined)
def paramsUpdateWithQDays(params, t, inf_rate, phi_s, g_rate, q_days, n, m):
    beta, beta_prime, phi_s_1, g_beta = -np.inf, -np.inf, -np.inf, -np.inf
    params_new = deepcopy(params)
    new_q_day = 14
    
    if t <= n:
        beta = inf_rate[0]
        beta_prime = beta * 1.2
        phi_s_1 = phi_s[0]
        g_beta = g_rate[0]
        new_q_day = q_days[0]
    else:
        for i in range(len(inf_rate) - 1):
            if t > n + m * i and t <= n + m * (i + 1):
                beta = inf_rate[i + 1]
                beta_prime = beta * 1.2
                phi_s_1 = phi_s[i + 1]
                g_beta = g_rate[i + 1]
                new_q_day = q_days[i + 1]
                break
    
    if beta != -np.inf:
        params_new['beta'] = beta
        params_new['beta_prime'] = beta_prime
        params_new['phi_s_1'] = phi_s_1
        params_new['g_beta'] = g_beta

    return params_new, new_q_day


# daily dynamics of parameters (User Defined) (for parameter learning)
def paramsUpdateDaily(params, t, inf_rate, phi_s, g_rate, n, m):
    beta, beta_prime, phi_s_1, g_beta = -np.inf, -np.inf, -np.inf, -np.inf
    params_new = deepcopy(params)
    
    beta = inf_rate[min(int(t), len(inf_rate) - 1)]
    beta_prime = beta * 1.2
    phi_s_1 = phi_s[min(int(t), len(inf_rate) - 1)]
    g_beta = g_rate[min(int(t), len(inf_rate) - 1)]
    
    if beta != -np.inf:
        params_new['beta'] = beta
        params_new['beta_prime'] = beta_prime
        params_new['phi_s_1'] = phi_s_1
        params_new['g_beta'] = g_beta

    return params_new

# to format and sample whole number values
def sampleResult(yy, sample_num_factor, N_t):

    samp_index = [int(i) for i in np.linspace(0, N_t * sample_num_factor - 1, N_t)]
    sampled_yy = []

    for t in samp_index:
        sampled_yy.append(yy[t])

    return sampled_yy


# use this function to run the simulation for optimization
# return value: time-series evolution of each states
def runModel(init_cond, T_delay, params, duration, inf_rate, phi_s, g_rate):
    # increase number of samples for better accuracy
    sample_num_factor = 100


    history = lambda t: init_cond
    lags = T_delay
    N_t = duration
    parameters = (params, inf_rate, phi_s, g_rate)
    t_span = np.linspace(1, N_t, N_t * sample_num_factor)
    # print(t_span)


    t0 = time.time()
    # ddeint(function, history, timepoints, fargs=([delays], ))
    yy = ddeint(ddeModel, history, t_span, fargs=(lags, parameters))
    # print("Time Elapsed:", time.time() - t0)
    # print("================\n", np.array(yy).shape)
    yy = sampleResult(yy, sample_num_factor, N_t)
    # print("================\n", np.array(yy).shape)

    return yy

# to run the SIRTEM given all rates for a specific duration
def runSimulation(init_cond, T_delay, params, duration, inf_rate, phi_s, g_rate, q_days):
    
    # increase number of samples for better accuracy
    sample_num_factor = 100

    history = lambda t: init_cond
    lags = T_delay
    N_t = duration
    parameters = (params, inf_rate, phi_s, g_rate)
    t_span = np.linspace(1, N_t, N_t * sample_num_factor)
    # print(t_span)


    t0 = time.time()
    # ddeint(function, history, timepoints, fargs=([delays], ))
    yy = ddeint(ddeModel, history, t_span, fargs=(lags, parameters))
    print("Time Elapsed:", time.time() - t0)
    # print("================\n", np.array(yy).shape)
    yy = sampleResult(yy, sample_num_factor, N_t)
    # print("================\n", np.array(yy).shape)

    return yy
    
