from knnsearch import knnsearch
import numpy as np


def our_match(phiM, phiN):
    C = np.linalg.pinv(phiM)@phiN
    match = knnsearch(phiM@C, phiN)
    return match
  

def our_match_desc(phiM, phiN, descM, descN):
    F = np.linalg.pinv(phiM)@descM
    G = np.linalg.pinv(phiN)@descN
    C = F@np.linalg.pinv(G)
    match_1 = knnsearch(phiM@C, phiN)
    match_2 = knnsearch(descM, descN)
    return match_1, phiM@C, phiN, match_2
  

def our_match_sym(phiM, phiN, descM, descN, ind_0_source, ind_1_source, ind_0_target, ind_1_target):

    match_same =  np.zeros((1,1000))
    match_opposite = np.zeros((1,1000))
    F = np.linalg.pinv(phiM)@descM
    G = np.linalg.pinv(phiN)@descN
    C = F@np.linalg.pinv(G)

    match_0 = knnsearch(phiM[ind_0_source,:]@C, phiN[ind_0_target,:])
    match_1 = knnsearch(phiM[ind_1_source,:]@C, phiN[ind_1_target,:])
    tar_act_0 = ind_0_target[match_0]
    tar_act_1 = ind_1_target[match_1]
    match_same[0,ind_0_source] = tar_act_0
    match_same[0,ind_1_source] = tar_act_1
 
    match_0 = knnsearch(phiM[ind_0_source,:]@C, phiN[ind_1_target,:])
    match_1 = knnsearch(phiM[ind_1_source,:]@C, phiN[ind_0_target,:])
    tar_act_0 = ind_1_target[match_0]
    tar_act_1 = ind_0_target[match_1]
    match_opposite[0,ind_0_source] = tar_act_0
    match_opposite[0,ind_1_source] = tar_act_1


    return match_same, match_opposite
