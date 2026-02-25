#! /usr/bin/env python3

import numpy as np
import math as m
"""
    # {daniele priola}
    # {priola@kth.se}
"""

l0 = 0.07
l1 = 0.3
l2 = 0.35

def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]
    q = [0.0, 0.0, 0.0]

    """
    Fill in your IK solution here and return the three joint values in q
    """
    x = x-l0
    
    q[2] = z
    r = np.sqrt(x**2+y**2)
    cos_di_q2 = (x**2+y**2-l1**2-l2**2)/(2*l1*l2)
    if cos_di_q2 > 1:
        cos_di_q2 = 1
    elif cos_di_q2 < -1:
        cos_di_q2 = -1

    q[1] = np.arctan2(np.sqrt(1-cos_di_q2**2),cos_di_q2)
    q[0] = np.arctan2(y,x) - np.arctan2(l2 * np.sin(q[1]),l1 + (l2 * np.cos(q[1])))
    
    
    return q


def kuka_IK(point, R, joint_positions):
    point = np.array(point,dtype = float).reshape(3)
    x = point[0]
    y = point[1]
    z = point[2]

    print("step 1")
    q = np.array(joint_positions, dtype = float)
    R = np.array(R)

    print("step 2")
    iterazioni_massime = 100
    tolleranza = 0.01
    j = 0
    while j < iterazioni_massime:
       
        print("step 3")
        T1 = forward_kinematic_kuka(q[0],(m.pi/2), 0.311, 0)
        T2 = forward_kinematic_kuka(q[1],-(m.pi/2), 0, 0)
        T3 = forward_kinematic_kuka(q[2],-(m.pi/2), 0.4, 0)
        T4 = forward_kinematic_kuka(q[3],(m.pi/2), 0, 0)
        T5 = forward_kinematic_kuka(q[4],(m.pi/2), 0.39, 0)
        T6 = forward_kinematic_kuka(q[5],-(m.pi/2), 0, 0)
        T7 = forward_kinematic_kuka(q[6],0,0, 0)
        print("step 4")

        T_tot = T1@T2@T3@T4@T5@T6@T7
        T_endeffector = T_tot @ np.array([0,0,0.078,1]) 
        posizione_corrente = T_endeffector[:3]
        
        print("step 5")
        errore_posizione = point - posizione_corrente
        
        errori_angoli = (0.5)*(np.cross(T_tot[:3, 0], R[:,0]) + np.cross(T_tot[:3,1], R[:,1]) + np.cross(T_tot[:3,2], R[:,2]))
        errori_entrambi = np.concatenate((errore_posizione, errori_angoli))

        nuova_tolleranza = np.linalg.norm(errori_entrambi)  
        if nuova_tolleranza < tolleranza:
            break
        print("step 6")  
            
        jacob = []
        jacob.append(trova_jacobiana_colonna(posizione_corrente, [0,0,0], [0,0,1]))
        
        tutte_T = [T1, T1@T2, T1@T2@T3, T1@T2@T3@T4, T1@T2@T3@T4@T5, T1@T2@T3@T4@T5@T6]
        for T_n in tutte_T:
            z = (T_n)[:3,2]
            p = (T_n)[:3,3]
            jacob.append(trova_jacobiana_colonna(posizione_corrente, z, p))

        print("step 7")
        jacobiana  = np.column_stack(jacob)
        print("step 8")
        
        pseudo_jacobiana  = np.linalg.pinv(jacobiana)
        variazione_giunture = pseudo_jacobiana @ errori_entrambi  

        q = q + variazione_giunture
        print("step 9")
        j = j + 1

    return q


def forward_kinematic_kuka(thetha, alpha, di, a):
    
    T = np.empty((4,4))
    T.fill(0)

    T[0][0] = np.cos(thetha)
    T[0][1] = -(np.sin(thetha))*(np.cos(alpha))
    T[0][2] = (np.sin(thetha))*(np.sin(alpha))
    T[0][3] = a*(np.cos(thetha))
    T[1][0] = np.sin(thetha)
    T[1][1] = (np.cos(thetha))*(np.cos(alpha))
    T[1][2] = -(np.cos(thetha))*(np.sin(alpha))
    T[1][3] = a*(np.sin(thetha))
    T[2][0] = 0
    T[2][1] = np.sin(alpha)
    T[2][2] = np.cos(alpha)
    T[2][3] = di
    T[3][0] = 0
    T[3][1] = 0 
    T[3][2] = 0
    T[3][3] = 1
    print("step foward kin")
    return np.array(T)

def trova_jacobiana_colonna(p_effector,z,p_effetor_finale):

    print("step jacob 1")
    p_effetor_finale = np.asarray(p_effetor_finale).reshape(3)
    p_effector = np.asarray(p_effector).reshape(3)
    z = np.asarray(z).reshape(3)

    print("step jacob 2")
    JacobV = np.cross(z, p_effector- p_effetor_finale)  
    JacobW = z 
    
    print("step jacob 3")
    Jac = []
    Jac = np.concatenate((JacobV,JacobW))
    
    return Jac
