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
import ground_state as gs


############################################################################
def get_d9Ld9L_state_indices(VS,d_double,S_val, Sz_val):
    '''
    Get d8L state index
    one hole must be dz2 corresponding to s electron
    
    Choose interesing states to plot spectra
    see George's email on Aug.21, 2021
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d9Ld9L_a1L_b1L_state_indices = []; d9Ld9L_a1L_b1L_state_labels = []
    d9Ld9L_b1L_a1L_state_indices = []; d9Ld9L_b1L_a1L_state_labels = []
    
    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']        
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        o4 = state['hole4_orb']        
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        slabel=[s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4]
        slabel= gs.make_z_canonical(slabel)
        s1 = slabel[0]; o1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
        s2 = slabel[5]; o2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
        s3 = slabel[10]; o3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
        s4 = slabel[15]; o4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];            

 
        if not ((o1 in pam.Ni_Cu_orbs) and (o2 in pam.Ni_Cu_orbs) and (o3 in pam.O_orbs) and (o4 in pam.O_orbs) \
                and z1==z3==1 and z2==z4==0):
            continue    
            
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3] 
            
            
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]


        # d9_{a1b1} singlet:
        if S12==0 and Sz12==0 and o1=='d3z2r2' and o2=='dx2y2' and o3=='px' and o4=='px' and s1=='up' and s2=='up' and s3=='dn' and s4=='dn':
            d9Ld9L_a1L_b1L_state_indices.append(i); d9Ld9L_a1L_b1L_state_labels.append('$S=0,a1L-b1L$')
            print ("a1L_b1L_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3,s4,o4,x4, y4, z4)

        # d9_{b1a1} singlet:
        if S12==0 and Sz12==0 and o1=='dx2y2' and o2=='d3z2r2' and o3=='px' and o4=='px'  and s1=='up' and s2=='up' and s3=='dn' and s4=='dn':
            d9Ld9L_b1L_a1L_state_indices.append(i); d9Ld9L_b1L_a1L_state_labels.append('$S=0,a1L-b1L$')
            print ("b1L_a1L_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3,s4,o4,x4, y4, z4)         

    return d9Ld9L_a1L_b1L_state_indices, d9Ld9L_a1L_b1L_state_labels, \
           d9Ld9L_b1L_a1L_state_indices, d9Ld9L_b1L_a1L_state_labels

def get_d9d9L2_state_indices(VS,d_double,S_val, Sz_val):
    '''
    Get d8L state index
    one hole must be dz2 corresponding to s electron
    
    Choose interesing states to plot spectra
    see George's email on Aug.21, 2021
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d9d9L2_a1_b1L2_state_indices = []; d9d9L2_a1_b1L2_state_labels = []
    d9d9L2_b1_a1L2_state_indices = []; d9d9L2_b1_a1L2_state_labels = []
    
    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']        
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        o4 = state['hole4_orb']        
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        slabel=[s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4]
        slabel= gs.make_z_canonical(slabel)
        s1 = slabel[0]; o1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
        s2 = slabel[5]; o2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
        s3 = slabel[10]; o3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
        s4 = slabel[15]; o4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];            

 
        if not ((o1 in pam.Ni_Cu_orbs) and (o2 in pam.Ni_Cu_orbs) and (o3 in pam.O_orbs) and (o4 in pam.O_orbs) \
                and z1==1 and z2==z3==z4==0):
            continue    
            
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3] 
            
            
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]


        # d9_{a1b1} singlet:
        if S12==0 and Sz12==0 and o1=='d3z2r2' and o2=='dx2y2' and o3=='px' and o4=='px' and s1=='up' and s2=='up' and s3=='dn' and s4=='dn':
            d9d9L2_a1_b1L2_state_indices.append(i); d9d9L2_a1_b1L2_state_labels.append('$S=0,a1-b1L^{2}$')
            print ("a1_b1L2_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3,s4,o4,x4, y4, z4)

        # d9_{b1a1} singlet:
        if S12==0 and Sz12==0 and o1=='dx2y2' and o2=='d3z2r2' and o3=='px' and o4=='px'  and s1=='up' and s2=='up' and s3=='dn' and s4=='dn':
            d9d9L2_b1_a1L2_state_indices.append(i); d9d9L2_b1_a1L2_state_labels.append('$S=0,a1-b1L^{2}$')
            print ("b1_a1L2_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3,s4,o4,x4, y4, z4)         

    return d9d9L2_a1_b1L2_state_indices, d9d9L2_a1_b1L2_state_labels, \
           d9d9L2_b1_a1L2_state_indices, d9d9L2_b1_a1L2_state_labels


def get_d9L2d9_state_indices(VS,d_double,S_val, Sz_val):
    '''
    Get d8L state index
    one hole must be dz2 corresponding to s electron
    
    Choose interesing states to plot spectra
    see George's email on Aug.21, 2021
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d9L2d9_a1L2_b1_state_indices = []; d9L2d9_a1L2_b1_state_labels = []
    d9L2d9_b1L2_a1_state_indices = []; d9L2d9_b1L2_a1_state_labels = []
    
    for i in range(0,dim):
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']        
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        o4 = state['hole4_orb']        
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        slabel=[s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4]
        slabel= gs.make_z_canonical(slabel)
        s1 = slabel[0]; o1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
        s2 = slabel[5]; o2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
        s3 = slabel[10]; o3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
        s4 = slabel[15]; o4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];            

 
        if not ((o1 in pam.Ni_Cu_orbs) and (o2 in pam.Ni_Cu_orbs) and (o3 in pam.O_orbs) and (o4 in pam.O_orbs) \
                and z1==z3==z4==1 and z2==0):
            continue    
            
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3] 
            
            
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]


        # d9_{a1b1} singlet:
        if S12==0 and Sz12==0 and o1=='d3z2r2' and o2=='dx2y2' and o3=='px' and o4=='px' and s1=='up' and s2=='up' and s3=='dn' and s4=='dn':
            d9L2d9_a1L2_b1_state_indices.append(i); d9L2d9_a1L2_b1_state_labels.append('$S=0,a1L^{2}-b1$')
            print ("a1L2_b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3,s4,o4,x4, y4, z4)

        # d9_{b1a1} singlet:
        if S12==0 and Sz12==0 and o1=='dx2y2' and o2=='d3z2r2' and o3=='px' and o4=='px'  and s1=='up' and s2=='up' and s3=='dn' and s4=='dn':
            d9L2d9_b1L2_a1_state_indices.append(i); d9L2d9_b1L2_a1_state_labels.append('$S=0,a1L^{2}-b1$')
            print ("b1L2_a1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3,s4,o4,x4, y4, z4)         

    return d9L2d9_a1L2_b1_state_indices, d9L2d9_a1L2_b1_state_labels, \
           d9L2d9_b1L2_a1_state_indices, d9L2d9_b1L2_a1_state_labels


def get_d8d9L_state_indices(VS,sym,d_double,S_val, Sz_val, AorB_sym, ANi):
    '''
    Get d8L state index
    one hole must be dz2 corresponding to s electron
    
    Choose interesing states to plot spectra
    see George's email on Aug.21, 2021
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d8d9L_state_indices= []; d8d9L_state_labels= []
 
    # get info specific for particular sym
    state_order, interaction_mat, Stot, Sz_set, AorB = ham.get_interaction_mat(ANi, sym)
    sym_orbs = state_order.keys()
    
    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']        
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        o4 = state['hole4_orb']        
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        slabel=[s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4]
        slabel= gs.make_z_canonical(slabel)
        s1 = slabel[0]; o1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
        s2 = slabel[5]; o2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
        s3 = slabel[10]; o3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
        s4 = slabel[15]; o4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];            
               
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]
            
            
        if not ((o1 in pam.Ni_Cu_orbs) and (o2 in pam.Ni_Cu_orbs) and (o3 in pam.Ni_Cu_orbs) and (o4 in pam.O_orbs) \
                and z1==z2==1 and z3==z4==0):
            continue    
            
        o12 = sorted([o1,o2])
        o12 = tuple(o12)    
                     
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3]
            
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]

        # for triplet, only need one Sz state; other Sz states have the same A(w)
       # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':

    
        if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
            continue
            
        if 'dx2y2' not in o12:
            continue
            
#         if o1==o2=='dxz' or o1==o2=='dyz':
#             if AorB_sym[i]!=AorB:
#                 continue

        if not o3=="dx2y2":
            continue
        

        d8d9L_state_indices.append(i)
       

    return d8d9L_state_indices


def get_d8Ld9_state_indices(VS,sym,d_double,S_val, Sz_val, AorB_sym, ANi):
    '''
    Get d8L state index
    one hole must be dz2 corresponding to s electron
    
    Choose interesing states to plot spectra
    see George's email on Aug.21, 2021
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d8Ld9_state_indices= []; d8Ld9_state_labels= []
 
    # get info specific for particular sym
    state_order, interaction_mat, Stot, Sz_set, AorB = ham.get_interaction_mat(ANi, sym)
    sym_orbs = state_order.keys()
    
    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']        
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        o4 = state['hole4_orb']        
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        slabel=[s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4]
        slabel= gs.make_z_canonical(slabel)
        s1 = slabel[0]; o1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
        s2 = slabel[5]; o2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
        s3 = slabel[10]; o3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
        s4 = slabel[15]; o4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];            
               
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]
            
            
        if not ((o1 in pam.Ni_Cu_orbs) and (o2 in pam.Ni_Cu_orbs) and (o3 in pam.Ni_Cu_orbs) and (o4 in pam.O_orbs) \
                and z1==z2==z4==1 and z3==0):
            continue    
            
        o12 = sorted([o1,o2])
        o12 = tuple(o12)    
                     
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3]
            
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]

        # for triplet, only need one Sz state; other Sz states have the same A(w)
       # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':

    
        if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
            continue
            
        if 'dx2y2' not in o12:
            continue
            
#         if o1==o2=='dxz' or o1==o2=='dyz':
#             if AorB_sym[i]!=AorB:
#                 continue

        if not o3=="dx2y2":
            continue
        

        d8Ld9_state_indices.append(i)
       

    return d8Ld9_state_indices


def get_d8d8_state_indices(VS,sym,d_double,S_val, Sz_val, AorB_sym, ANi):
    '''
    Get d8L state index
    one hole must be dz2 corresponding to s electron
    
    Choose interesing states to plot spectra
    see George's email on Aug.21, 2021
    '''    
    Norb = pam.Norb
    dim = VS.dim
    d8d8_state_indices= []; d8d8_state_labels= []
 
    # get info specific for particular sym
    state_order, interaction_mat, Stot, Sz_set, AorB = ham.get_interaction_mat(ANi, sym)
    sym_orbs = state_order.keys()
    
    for i in d_double:
        # state is original state but its orbital info remains after basis change
        state = VS.get_state(VS.lookup_tbl[i])
            
        s1 = state['hole1_spin']
        s2 = state['hole2_spin']
        s3 = state['hole3_spin']
        s4 = state['hole4_spin']        
        o1 = state['hole1_orb']
        o2 = state['hole2_orb']
        o3 = state['hole3_orb']
        o4 = state['hole4_orb']        
        x1, y1, z1 = state['hole1_coord']
        x2, y2, z2 = state['hole2_coord']
        x3, y3, z3 = state['hole3_coord']
        x4, y4, z4 = state['hole4_coord']
        slabel=[s1,o1,x1,y1,z1,s2,o2,x2,y2,z2,s3,o3,x3,y3,z3,s4,o4,x4,y4,z4]
        slabel= gs.make_z_canonical(slabel)
        s1 = slabel[0]; o1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
        s2 = slabel[5]; o2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
        s3 = slabel[10]; o3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
        s4 = slabel[15]; o4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];            
               
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]
            
            
        if not ((o1 in pam.Ni_Cu_orbs) and (o2 in pam.Ni_Cu_orbs) and (o3 in pam.Ni_Cu_orbs) and (o4 in pam.Ni_Cu_orbs) \
                and z1==z2==1 and z3==z4==0):
            continue    
            
        o12 = sorted([o1,o2])
        o12 = tuple(o12)    
                     
        orbs = [o1,o2,o3]
        xs = [x1,x2,x3]
        ys = [y1,y2,y3]
        zs = [z1,z2,z3]
            
        # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
        S12  = S_val[i]
        Sz12 = Sz_val[i]

        # for triplet, only need one Sz state; other Sz states have the same A(w)
       # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':

    
        if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
            continue
            
        if 'dx2y2' not in o12:
            continue
            
#         if o1==o2=='dxz' or o1==o2=='dyz':
#             if AorB_sym[i]!=AorB:
#                 continue

        if  o3!="dx2y2" and o4!="dx2y2":
            continue
        

        d8d8_state_indices.append(i)
       

    return d8d8_state_indices


# 0
#             continue
            
#         orbs = [o1,o2,o3]
#         xs = [x1,x2,x3]
#         ys = [y1,y2,y3]
#         zs = [z1,z2,z3]
#         idx = orbs.index(porbs[0])
        
            
#         if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
#             continue
            
#         # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         # for triplet, only need one Sz state; other Sz states have the same A(w)
#        # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':
#         dorbs = sorted(dorbs)
        
#         # d8_{a1b1} singlet:
#         if S12==0 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d8d10L_a1b1_S0_state_indices.append(i); d8d10L_a1b1_S0_state_labels.append('$S=0,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)

#         # d8_{a1b1} triplet:
#         if S12==1 and Sz12==1 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d8d10L_a1b1_S1_Sz1_state_indices.append(i); d8d10L_a1b1_S1_Sz1_state_labels.append('$S=1,S_z=1,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)            
        
#         if S12==1 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d8d10L_a1b1_S1_Sz0_state_indices.append(i); d8d10L_a1b1_S1_Sz0_state_labels.append('$S=1,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)         

#         if S12==1 and Sz12==-1 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d8d10L_a1b1_S1_Sz_1_state_indices.append(i); d8d10L_a1b1_S1_Sz_1_state_labels.append('$S=1,S_z=-1,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)             
            
#         # d8_{b1b1} singlet:
#         if S12==0 and Sz12==0 and dorbs[0]=='dx2y2' and dorbs[1]=='dx2y2':
#             d8d10L_b1b1_state_indices.append(i); d8d10L_b1b1_state_labels.append('$S=0,S_z=0,d^8_{x^2-y^2,x^2-y^2}Ls$')
#             print ("b1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)            

#     return d8d10L_a1b1_S0_state_indices, d8d10L_a1b1_S0_state_labels, \
#            d8d10L_a1b1_S1_Sz1_state_indices, d8d10L_a1b1_S1_Sz1_state_labels, \
#            d8d10L_a1b1_S1_Sz0_state_indices, d8d10L_a1b1_S1_Sz0_state_labels, \
#            d8d10L_a1b1_S1_Sz_1_state_indices, d8d10L_a1b1_S1_Sz_1_state_labels, \
#            d8d10L_b1b1_state_indices,d8d10L_b1b1_state_labels

# def get_d8Ld10_state_indices(VS,d_double,S_val, Sz_val):
#     '''
#     Get d8L state index
#     one hole must be dz2 corresponding to s electron
    
#     Choose interesing states to plot spectra
#     see George's email on Aug.21, 2021
#     '''    
#     Norb = pam.Norb
#     dim = VS.dim
#     d8Ld10_a1b1_S0_state_indices= []; d8Ld10_a1b1_S0_state_labels= []
    
#     for i in d_double:
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
            
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         o1 = state['hole1_orb']
#         o2 = state['hole2_orb']
#         o3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']
            
#         nNi_Cu, nO, dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
#         if not (nNi_Cu==2 and nO==1):
#             continue
            
#         if not z1==z2==z3==1:
#             continue
            
#         orbs = [o1,o2,o3]
#         xs = [x1,x2,x3]
#         ys = [y1,y2,y3]
#         zs = [z1,z2,z3]
#         idx = orbs.index(porbs[0])
        
            
#         if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
#             continue
            
#         # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         # for triplet, only need one Sz state; other Sz states have the same A(w)
#        # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':
#         dorbs = sorted(dorbs)
        
#         # d8_{a1b1} singlet:
#         if S12==0 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d8Ld10_a1b1_S0_state_indices.append(i); d8Ld10_a1b1_S0_state_labels.append('$S=0,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)          

#     return d8Ld10_a1b1_S0_state_indices, d8Ld10_a1b1_S0_state_labels

# def get_d10Ld8_state_indices(VS,d_double,S_val, Sz_val):
#     '''
#     Get d8L state index
#     one hole must be dz2 corresponding to s electron
    
#     Choose interesing states to plot spectra
#     see George's email on Aug.21, 2021
#     '''    
#     Norb = pam.Norb
#     dim = VS.dim
#     d10Ld8_a1b1_S0_state_indices= []; d10Ld8_a1b1_S0_state_labels= []
#     d10Ld8_a1b1_S1_Sz1_state_indices= []; d10Ld8_a1b1_S1_Sz1_state_labels= []
#     d10Ld8_a1b1_S1_Sz0_state_indices= []; d10Ld8_a1b1_S1_Sz0_state_labels= []
#     d10Ld8_a1b1_S1_Sz_1_state_indices= []; d10Ld8_a1b1_S1_Sz_1_state_labels= []
#     d10Ld8_b1b1_state_indices= [];d10Ld8_b1b1_state_labels= []
    
#     for i in d_double:
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
            
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         o1 = state['hole1_orb']
#         o2 = state['hole2_orb']
#         o3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']
            
#         nNi_Cu, nO, dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
#         if not (nNi_Cu==2 and nO==1):
#             continue
            
#         if not ((o1 in pam.O_orbs and z1==1 and z2==z3==0) or (o2 in pam.O_orbs and z2==1 and z1==z3==0) or (o3 in pam.O_orbs and z3==1 and z1==z2==0)):
#             continue
            
#         orbs = [o1,o2,o3]
#         xs = [x1,x2,x3]
#         ys = [y1,y2,y3]
#         zs = [z1,z2,z3]
#         idx = orbs.index(porbs[0])
        
            
#         if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
#             continue
            
#         # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         # for triplet, only need one Sz state; other Sz states have the same A(w)
#        # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':
#         dorbs = sorted(dorbs)
        
#         # d8_{a1b1} singlet:
#         if S12==0 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d10Ld8_a1b1_S0_state_indices.append(i); d10Ld8_a1b1_S0_state_labels.append('$S=0,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)

#         # d8_{a1b1} triplet:
#         if S12==1 and Sz12==1 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d10Ld8_a1b1_S1_Sz1_state_indices.append(i); d10Ld8_a1b1_S1_Sz1_state_labels.append('$S=1,S_z=1,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)            
        
#         if S12==1 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d10Ld8_a1b1_S1_Sz0_state_indices.append(i); d10Ld8_a1b1_S1_Sz0_state_labels.append('$S=1,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)         

#         if S12==1 and Sz12==-1 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d10Ld8_a1b1_S1_Sz_1_state_indices.append(i); d10Ld8_a1b1_S1_Sz_1_state_labels.append('$S=1,S_z=-1,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)             
            
#         # d8_{b1b1} singlet:
#         if S12==0 and Sz12==0 and dorbs[0]=='dx2y2' and dorbs[1]=='dx2y2':
#             d10Ld8_b1b1_state_indices.append(i); d10Ld8_b1b1_state_labels.append('$S=0,S_z=0,d^8_{x^2-y^2,x^2-y^2}Ls$')
#             print ("b1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)            

#     return d10Ld8_a1b1_S0_state_indices, d10Ld8_a1b1_S0_state_labels, \
#            d10Ld8_a1b1_S1_Sz1_state_indices, d10Ld8_a1b1_S1_Sz1_state_labels, \
#            d10Ld8_a1b1_S1_Sz0_state_indices, d10Ld8_a1b1_S1_Sz0_state_labels, \
#            d10Ld8_a1b1_S1_Sz_1_state_indices, d10Ld8_a1b1_S1_Sz_1_state_labels, \
#            d10Ld8_b1b1_state_indices,d10Ld8_b1b1_state_labels

# def get_d10d8L_state_indices(VS,d_double,S_val, Sz_val):
#     '''
#     Get d8L state index
#     one hole must be dz2 corresponding to s electron
    
#     Choose interesing states to plot spectra
#     see George's email on Aug.21, 2021
#     '''    
#     Norb = pam.Norb
#     dim = VS.dim
#     d10d8L_a1b1_S0_state_indices= []; d10d8L_a1b1_S0_state_labels= []
    
#     for i in d_double:
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
            
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         o1 = state['hole1_orb']
#         o2 = state['hole2_orb']
#         o3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']
            
#         nNi_Cu, nO, dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
#         if not (nNi_Cu==2 and nO==1):
#             continue
            
#         if not z1==z2==z3==0:
#             continue
            
#         orbs = [o1,o2,o3]
#         xs = [x1,x2,x3]
#         ys = [y1,y2,y3]
#         zs = [z1,z2,z3]
#         idx = orbs.index(porbs[0])
        
            
#         if not (abs(xs[idx]-1.0)<1.e-3 and abs(ys[idx])<1.e-3):
#             continue
            
#         # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         # for triplet, only need one Sz state; other Sz states have the same A(w)
#        # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':
#         dorbs = sorted(dorbs)
        
#         # d8_{a1b1} singlet:
#         if S12==0 and Sz12==0 and dorbs[0]=='d3z2r2' and dorbs[1]=='dx2y2':
#             d10d8L_a1b1_S0_state_indices.append(i); d10d8L_a1b1_S0_state_labels.append('$S=0,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)          

#     return d10d8L_a1b1_S0_state_indices, d10d8L_a1b1_S0_state_labels

# def get_d8d9_state_indices(VS,sym,d_double,S_val, Sz_val, AorB_sym, ANi):
#     '''
#     Get d8L state index
#     one hole must be dz2 corresponding to s electron
    
#     Choose interesing states to plot spectra
#     see George's email on Aug.21, 2021
#     '''    
#     Norb = pam.Norb
#     dim = VS.dim
#     d8d9_state_indices= []; d8d9_state_labels= []

#     Nidorbs= []; Cudorbs= []
    
    
#     # get info specific for particular sym
#     state_order, interaction_mat, Stot, Sz_set, AorB = ham.get_interaction_mat(ANi, sym)
#     sym_orbs = state_order.keys()
    
#     for i in d_double:
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
            
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         o1 = state['hole1_orb']
#         o2 = state['hole2_orb']
#         o3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']
               
#         # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]
            
#         nNi_Cu, nO, dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
#         if not (nNi_Cu==3):
#             continue
            
#         if not ((z1==0 and z2==z3==1) or (z2==0 and z1==z3==1) or (z3==0 and z1==z2==1)):
#             continue
           

#         if z2==z3==1:
#             o12 = sorted([o2,o3])
#             o12 = tuple(o12)
#             Cud = o1
#         elif z1==z3==1:
#             o12 = sorted([o1,o3])
#             o12 = tuple(o12)
#             Cud = o2
#         elif z1==z2==1:
#             o12 = sorted([o1,o2])
#             o12 = tuple(o12)
#             Cud = o3

                     
#         orbs = [o1,o2,o3]
#         xs = [x1,x2,x3]
#         ys = [y1,y2,y3]
#         zs = [z1,z2,z3]
            
#         # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         # for triplet, only need one Sz state; other Sz states have the same A(w)
#        # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':
#         dorbs = sorted(dorbs)
# #         print (o1,x1,y1,z1,o2,x2,y2,z2,o3,x3,y3,z3)
#         if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
#             continue
            
#         if 'dx2y2' not in o12:
#             continue
            
# #         if o1==o2=='dxz' or o1==o2=='dyz':
# #             if AorB_sym[i]!=AorB:
# #                 continue

#         if not Cud=="dx2y2":
#             continue
        

#         d8d9_state_indices.append(i)
        

#     return d8d9_state_indices



# def get_d9d8_state_indices(VS,sym,d_double,S_val, Sz_val, AorB_sym, ACu):
#     '''
#     Get d8L state index
#     one hole must be dz2 corresponding to s electron
    
#     Choose interesing states to plot spectra
#     see George's email on Aug.21, 2021
#     '''    
#     Norb = pam.Norb
#     dim = VS.dim
#     d9d8_a1b1a1_state_indices= []; d9d8_a1b1a1_state_labels= []
#     d9d8_a1b1b1_state_indices= []; d9d8_a1b1b1_state_labels= []
#     d9d8_a1a1b1_state_indices= []; d9d8_a1a1b1_state_labels= []
#     Nidorbs= []; Cudorbs= []
    
    
#     # get info specific for particular sym
#     state_order, interaction_mat, Stot, Sz_set, AorB = ham.get_interaction_mat(ACu, sym)
#     sym_orbs = state_order.keys()
    
#     for i in d_double:
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
            
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         o1 = state['hole1_orb']
#         o2 = state['hole2_orb']
#         o3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']
               
#         # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]
            
#         nNi_Cu, nO, dorbs, porbs = util.get_statistic_3orb(o1,o2,o3)
            
#         if not (nNi_Cu==3):
#             continue
            
#         if not ((z1==1 and z2==z3==0) or (z2==1 and z1==z3==0) or (z3==1 and z1==z2==0)):
#             continue
           

#         if z2==z3==0:
#             o12 = sorted([o2,o3])
#             o12 = tuple(o12)
#             Cud = o1
#         elif z1==z3==0:
#             o12 = sorted([o1,o3])
#             o12 = tuple(o12)
#             Cud = o2
#         elif z1==z2==0:
#             o12 = sorted([o1,o2])
#             o12 = tuple(o12)
#             Cud = o3

                     
#         orbs = [o1,o2,o3]
#         xs = [x1,x2,x3]
#         ys = [y1,y2,y3]
#         zs = [z1,z2,z3]
            
#         # S_val, Sz_val obtained from basis.create_singlet_triplet_basis_change_matrix
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         # for triplet, only need one Sz state; other Sz states have the same A(w)
#        # if Sz12==0 and 'd3z2r2' in dorbs and se=='up':
#         dorbs = sorted(dorbs)
# #         print (o1,x1,y1,z1,o2,x2,y2,z2,o3,x3,y3,z3,o12,Cud,o12[0],o12[1])
#         if o12 not in sym_orbs or S12!=Stot or Sz12 not in Sz_set:
#             continue
        
#         # d8_{a1b1a1} singlet:
#         if S12==0 and Sz12==0 and o12[0]=='d3z2r2' and o12[1]=='dx2y2' and Cud=='d3z2r2':
#             d9d8_a1b1a1_state_indices.append(i); d9d8_a1b1a1_state_labels.append('$S=0,S_z=0,d^8_{z^2,x^2-y^2,z^2}Ls$')
#             print ("a1b1a1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)

#         # d8_{a1b1b1} singlet:
#         if S12==0 and Sz12==0 and o12[0]=='d3z2r2' and o12[1]=='dx2y2' and Cud=='dx2y2':
#             d9d8_a1b1b1_state_indices.append(i); d9d8_a1b1b1_state_labels.append('$S=0,S_z=0,d^8_{z^2,x^2-y^2}Ls$')
#             print ("a1b1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)                 
            
#         # d8_{a1a1b1} singlet:
#         if S12==0 and Sz12==0 and o12[0]=='d3z2r2' and o12[1]=='d3z2r2' and Cud=='dx2y2':
#             d9d8_a1a1b1_state_indices.append(i);  d9d8_a1a1b1_state_labels.append('$S=0,S_z=0,d^8_{x^2-y^2,x^2-y^2}Ls$')
#             print ("b1b1_state_indices", i, ", state: orb= ",s1,o1,x1, y1, z1,s2,o2,x2, y2, z2,s3,o3,x3, y3, z3)            

#     return d9d8_a1b1a1_state_indices, d9d8_a1b1a1_state_labels, \
#            d9d8_a1b1b1_state_indices, d9d8_a1b1b1_state_labels, \
#            d9d8_a1a1b1_state_indices, d9d8_a1a1b1_state_labels