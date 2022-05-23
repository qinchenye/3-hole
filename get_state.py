'''
Get state index for A(w)
'''
import subprocess
import os
import sys
import time
import shutil
import numpy

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util

#####################################################################


############################################################################
def get_d8L_state_indices(VS,d_double,S_val, Sz_val):
    '''
    Get d8L state index
    one hole must be dz2 corresponding to s electron
    
    Choose interesing states to plot spectra
    see George's email on Aug.21, 2021
    '''    
    Norb = pam.Norb
    dim = VS.dim
    a1b1_S0_state_indices = []; a1b1_S0_state_labels = []
    a1b1_S1_state_indices = []; a1b1_S1_state_labels = []
    a1a1_state_indices = []; a1a1_state_labels = []
    
    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
            
        nNi, nO, nCu , dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
        if not ((nNi==2 or nCu==2 or (nNi==1 and nCu==1)) and nO==1):
            continue
            
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3]
        idx = orbs.index(porbs[0])
        
            
        if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
            continue
            
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]

        # for triplet, only need one Sz state; other Sz states have the same A(w)
       # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':
        dorbs = sorted(dorbs)
        
        # d8_{a1b1} singlet:
        if S12==0 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
            a1b1_S0_state_indices.append(i); a1b1_S0_state_labels.append('$S=0,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
            print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)
                
        # d8_{a1b1} triplet:
        if S12==1 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
            a1b1_S1_state_indices.append(i); a1b1_S1_state_labels.append('$S=1,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
            print ("a1b1_state_indices", i, ", state: orb= ", s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)
        
        # d8_{a1a1} singlet:
        if S12==0 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='d3z2r2':
            a1a1_state_indices.append(i); a1a1_state_labels.append('$S=0,S_z=0,d^8_{z^2,z^2}Ls$')
            print ("a1a1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)

    return a1b1_S0_state_indices, a1b1_S0_state_labels, \
           a1b1_S1_state_indices, a1b1_S1_state_labels, \
           a1a1_state_indices, a1a1_state_labels

def get_d9L2_state_indices(VS):
    '''
    Get d9L2s state index
    one hole must be dz2 corresponding to s electron
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d9L2_state_indices = []; d9L2_state_labels = []
    
    for i in xrange(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
        itype = state['type']
        

        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']

        nNi, nO, nCu , dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
        if not ((nNi==1 or nCu==1) and nO==2):
            continue
            
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3]
        
            
        idx = orbs.index(porbs[0])
        if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
            continue
        idx = orbs.index(porbs[1])
        if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
            continue
            
        if dorbs[0]=='dx2y2' and se=='up':
            d9L2s_state_indices.append(i); d9L2_state_labels.append('$d^9_{x^2-y^2}L^2s$')
            print ("d9L2_state_indices", i, ", state: orb= ", o1,o2,o3)

    return d9L2_state_indices, d9L2_state_labels

