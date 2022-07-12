import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util
import lanczos

def reorder_z(slabel):
    '''
    reorder the s, orb, coord's labeling a state to prepare for generating its canonical state
    Useful for three hole case especially !!!
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    
    if orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs and z2>z1:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
    elif orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs and (z2==z1 or z2<z1):
        state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    elif orb1 in pam.Ni_Cu_orbs and orb2 in pam.O_orbs:
        state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs and z2>z1:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs and (z2==z1 or z2<z1): 
        state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
            
    return state_label
                
def make_z_canonical(slabel):
    
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14]
    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    tmp12 = reorder_z(tlabel)

    tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
    tmp23 = reorder_z(tlabel)

    tlabel = tmp12[0:5]+tmp23[0:5]
    tmp = reorder_z(tlabel)

    slabel = tmp+tmp23[5:10]
                
    return slabel


def get_ground_state(matrix, VS, S_val,Sz_val):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    t1 = time.time()
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     print ('H=')
#     print (M_dense)
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()                                                               #calculate atom limit
#     print ('lowest eigenvalue of H from np.linalg.eigh = ')
#     print (vals)
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    
    #get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,1):                                                                          #gai
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.01)

        wgt_d9d8 = np.zeros(6)
        wgt_d8d9 = np.zeros(6)
        wgt_d9d9L = np.zeros(6)        
        wgt_d9d10L2= np.zeros(6)
        wgt_d10d9L2= np.zeros(6)
        wgt_d9L2d10= np.zeros(6)   
        wgt_d10Ld9L= np.zeros(6)  
        wgt_d9Ld10L= np.zeros(6)
        wgt_d10L2d9= np.zeros(6)        
        wgt_d10d8L= np.zeros(6)
        wgt_d9Ld9 = np.zeros(6)
        wgt_d8d10L = np.zeros(6)        
        wgt_d8Ld10 = np.zeros(6) 
        wgt_d10Ld8 = np.zeros(6)  
        wgt_d10d10 = np.zeros(6)         
        sumweight=0
        sumweight1=0

        print ("Compute the weights in GS (lowest Aw peak)")
        #for i in indices[0]:
        for i in range(0,len(vecs[:,k])):
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            weight = abs(vecs[i,k])**2
            
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            s3 = state['hole3_spin']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']

            #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
            #    continue
            S12  = S_val[i]
            Sz12 = Sz_val[i]

            o12 = sorted([orb1,orb2,orb3])
            o12 = tuple(o12)

#             if i in indices[0]:
#                 print (' state ', i, orb1,s1,x1,y1,z1,orb2,s2,x2,y2,z2,orb3,s3,x3,y3,z3, \
#                        'S=',S12,'Sz=',Sz12,", weight = ", weight,o12)
            slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
            slabel= make_z_canonical(slabel)
            s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
            s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
            s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14]
            if i in indices[0]: 
                sumweight1=sumweight1+abs(vecs[i,k])**2
                print (' state ', i, orb1,s1,x1,y1,z1,orb2,s2,x2,y2,z2,orb3,s3,x3,y3,z3, \
                       'S=',S12,'Sz=',Sz12,", weight = ", weight,o12)        
            
                
            if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='dx2y2'  and z1==1 and z2==z3==0  and S12==0 and Sz12==0:
                 wgt_d9d8[0]+=abs(vecs[i,k])**2
            if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='d3z2r2'  and z1==1 and z2==z3==0  and S12==0 and Sz12==0:
                 wgt_d9d8[1]+=abs(vecs[i,k])**2    
            if orb1=='dx2y2' and  orb2=='d3z2r2'  and  orb3=='dx2y2'  and z1==1 and z2==z3==0  and S12==1:
                 wgt_d9d8[2]+=abs(vecs[i,k])**2   
            if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='dx2y2'  and z1==1 and z2==z3==0  and S12==0 and Sz12==0:
                 wgt_d9d8[3]+=abs(vecs[i,k])**2                      
                    
                    
            if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='dx2y2' and z1==z2==1 and z3==0 and S12==0 and Sz12==0:
                 wgt_d8d9[0]+=abs(vecs[i,k])**2
            if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='d3z2r2'  and z1==z2==1 and z3==0  and S12==0 and Sz12==0:
                 wgt_d8d9[1]+=abs(vecs[i,k])**2    
            if orb1=='dx2y2' and  orb2=='d3z2r2'  and  orb3=='dx2y2'  and z1==z2==1 and z3==0  and S12==0 and Sz12==0:
                 wgt_d8d9[2]+=abs(vecs[i,k])**2   
            if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='dx2y2'  and z1==z2==1 and z3==0  and S12==1:
                 wgt_d8d9[3]+=abs(vecs[i,k])**2                      
                    
            if orb1=='dx2y2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')  and z1==1 and z2==z3==0 :
                 wgt_d9d9L[0]+=abs(vecs[i,k])**2   
            if orb1=='dx2y2' and  orb2=='d3z2r2'  and  (orb3=='px' or orb3=='py')  and z1==1 and z2==z3==0 :
                 wgt_d9d9L[1]+=abs(vecs[i,k])**2               
            if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')  and z1==1 and z2==z3==0 :
                 wgt_d9d9L[2]+=abs(vecs[i,k])**2                       
                    
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='py')  or  (orb3=='px' and orb2=='py'))  and z1==1 and z2==z3==0 :
                 wgt_d9d10L2[0]+=abs(vecs[i,k])**2   
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z1==1 and z2==z3==0 :
                 wgt_d9d10L2[1]+=abs(vecs[i,k])**2 
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='py')  or  (orb3=='px' and orb2=='py'))  and z1==1 and z2==z3==0 :
                 wgt_d9d10L2[2]+=abs(vecs[i,k])**2                       
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z1==1 and z2==z3==0 :
                 wgt_d9d10L2[3]+=abs(vecs[i,k])**2   

            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z1==z2==z3==0 :
                 wgt_d10d9L2[0]+=abs(vecs[i,k])**2                 
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='py')  or  (orb2=='py' and orb3=='px'))  and z1==z2==z3==0 :
                 wgt_d10d9L2[1]+=abs(vecs[i,k])**2 
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z1==z2==z3==0 :
                 wgt_d10d9L2[2]+=abs(vecs[i,k])**2                 
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='py')  or  (orb2=='py' and orb3=='px'))  and z1==z2==z3==0 :
                 wgt_d10d9L2[3]+=abs(vecs[i,k])**2                     
                    
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z1==z2==z3==1 :
                 wgt_d9L2d10[0]+=abs(vecs[i,k])**2                 
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='py')  or  (orb2=='py' and orb3=='px'))  and z1==z2==z3==1 :
                 wgt_d9L2d10[1]+=abs(vecs[i,k])**2                     
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z1==z2==z3==1 :
                 wgt_d9L2d10[2]+=abs(vecs[i,k])**2                 
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='py')  or  (orb2=='py' and orb3=='px'))  and z1==z2==z3==1 :
                 wgt_d9L2d10[3]+=abs(vecs[i,k])**2                     
                    
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z2==1 and z1==z3==0 :
                 wgt_d10Ld9L[0]+=abs(vecs[i,k])**2                 
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='py')  or  (orb2=='py' and orb3=='px'))  and z2==1 and z1==z3==0 :
                 wgt_d10Ld9L[1]+=abs(vecs[i,k])**2  
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z2==1 and z1==z3==0 :
                 wgt_d10Ld9L[2]+=abs(vecs[i,k])**2                 
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='py')  or  (orb2=='py' and orb3=='px'))  and z2==1 and z1==z3==0 :
                 wgt_d10Ld9L[3]+=abs(vecs[i,k])**2                    
                    
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z3==0 and z1==z2==1 :
                 wgt_d9Ld10L[0]+=abs(vecs[i,k])**2                 
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='py')  or  (orb2=='py' and orb3=='px'))  and z3==0 and z1==z2==1 :
                 wgt_d9Ld10L[1]+=abs(vecs[i,k])**2  
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z3==0 and z1==z2==1 :
                 wgt_d9Ld10L[2]+=abs(vecs[i,k])**2                 
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='py')  or  (orb2=='py' and orb3=='px'))  and z3==0 and z1==z2==1 :
                 wgt_d9Ld10L[3]+=abs(vecs[i,k])**2                     
                    
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='py')  or  (orb3=='px' and orb2=='py'))  and z1==0 and z2==z3==1 :
                 wgt_d10L2d9[0]+=abs(vecs[i,k])**2                       
            if orb1=='dx2y2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z1==0 and z2==z3==1 :
                 wgt_d10L2d9[1]+=abs(vecs[i,k])**2 
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='py')  or  (orb3=='px' and orb2=='py'))  and z1==0 and z2==z3==1 :
                 wgt_d10L2d9[2]+=abs(vecs[i,k])**2                       
            if orb1=='d3z2r2' and  ((orb2=='px' and orb3=='px')  or  (orb2=='py' and orb3=='py'))  and z1==0 and z2==z3==1 :
                 wgt_d10L2d9[3]+=abs(vecs[i,k])**2 
                    
            if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')  and z1==z2==z3==0 and S12==1 :
                 wgt_d10d8L[0]+=abs(vecs[i,k])**2       
            if  orb1=='dx2y2' and  orb2=='d3z2r2'  and  (orb3=='px' or orb3=='py')  and z1==z2==z3==0 and S12==0 :
                 wgt_d10d8L[1]+=abs(vecs[i,k])**2                       
                    
                  
            if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')  and z1==z3==1 and z2==0 :
                 wgt_d9Ld9[0]+=abs(vecs[i,k])**2 
            if orb1=='dx2y2' and  orb2=='d3z2r2'  and  (orb3=='px' or orb3=='py')  and z1==z3==1 and z2==0 :
                 wgt_d9Ld9[1]+=abs(vecs[i,k])**2                     


            if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')  and z1==z2==1 and z3==0 and S12==1 :
                 wgt_d8d10L[0]+=abs(vecs[i,k])**2  
            if orb1=='dx2y2' and  orb2=='d3z2r2'  and  (orb3=='px' or orb3=='py')  and z1==z2==1 and z3==0 and S12==0 :
                 wgt_d8d10L[1]+=abs(vecs[i,k])**2                      
                    
            if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')  and z1==z2==z3==1  and S12==1 :
                 wgt_d8Ld10[0]+=abs(vecs[i,k])**2 
            if  orb1=='dx2y2' and  orb2=='d3z2r2' and  (orb3=='px' or orb3=='py')  and z1==z2==z3==1  and S12==0 :
                 wgt_d8Ld10[1]+=abs(vecs[i,k])**2                     
                    
            if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')  and z1==z2==0 and z3==1  and S12==1 :
                 wgt_d10Ld8[0]+=abs(vecs[i,k])**2      
            if orb1=='dx2y2' and  orb2=='d3z2r2'  and  (orb3=='px' or orb3=='py')  and z1==z2==0 and z3==1  and S12==0 :
                 wgt_d10Ld8[1]+=abs(vecs[i,k])**2                     
                    
            if (orb1=='px' or orb1=='py') and  (orb2=='px' or orb2=='py')  and  (orb3=='px' or orb3=='py') :
                 wgt_d10d10[0]+=abs(vecs[i,k])**2                       
                    
            sumweight=sumweight+abs(vecs[i,k])**2
        print ('sumweight=',sumweight)
        print ('sumweight1=',sumweight1)
        
        txt=open('dx2y2dx2y2dx2y2d9d8','a')                                  
        txt.write(str(wgt_d9d8[0])+'\n')
        txt.close()
        txt=open('dx2y2dx2y2d3z2r2d9d8','a')                                  
        txt.write(str(wgt_d9d8[1])+'\n')
        txt.close()
        txt=open('dx2y2d3z2r2dx2y2d9d8S1','a')                                  
        txt.write(str(wgt_d9d8[2])+'\n')
        txt.close()        
        txt=open('d3z2r2dx2y2dx2y2d9d8','a')                                  
        txt.write(str(wgt_d9d8[3])+'\n')
        txt.close()
        wgt_d9d8[4]=wgt_d9d8[0]+wgt_d9d8[1]+wgt_d9d8[2]+wgt_d9d8[3]
        txt=open('d9d8','a')                                  
        txt.write(str(wgt_d9d8[4])+'\n')
        txt.close()        
        
        txt=open('dx2y2dx2y2dx2y2d8d9','a')                                  
        txt.write(str(wgt_d8d9[0])+'\n')
        txt.close()
        txt=open('dx2y2dx2y2d3z2r2d8d9','a')                                  
        txt.write(str(wgt_d8d9[1])+'\n')
        txt.close()
        txt=open('dx2y2d3z2r2dx2y2d8d9S1','a')                                  
        txt.write(str(wgt_d8d9[2])+'\n')
        txt.close()        
        txt=open('d3z2r2dx2y2dx2y2d8d9','a')                                  
        txt.write(str(wgt_d8d9[3])+'\n')
        txt.close()  
        wgt_d8d9[4]=wgt_d8d9[0]+wgt_d8d9[1]+wgt_d8d9[2]+wgt_d8d9[3]
        txt=open('d8d9','a')                                  
        txt.write(str(wgt_d8d9[4])+'\n')
        txt.close()         
        
        txt=open('dx2y2dx2y2pxd9d9L','a')                                  
        txt.write(str(wgt_d9d9L[0])+'\n')
        txt.close()
        txt=open('dx2y2d3z2r2pxd9d9L','a')                                  
        txt.write(str(wgt_d9d9L[1])+'\n')
        txt.close()
        txt=open('d3z2r2dx2y2pxd9d9L','a')                                  
        txt.write(str(wgt_d9d9L[2])+'\n')
        txt.close() 
        wgt_d9d9L[3]=wgt_d9d9L[0]+wgt_d9d9L[1]+wgt_d9d9L[2]
        txt=open('d9d9L','a')                                  
        txt.write(str(wgt_d9d9L[3])+'\n')
        txt.close()           
        
        
        txt=open('dx2y2pxpxd9d10L2','a')                                  
        txt.write(str(wgt_d9d10L2[0])+'\n')
        txt.close()
        txt=open('dx2y2pxpyd9d10L2','a')                                  
        txt.write(str(wgt_d9d10L2[1])+'\n')
        txt.close()
        txt=open('d3z2r2pxpyd9d10L2','a')                                  
        txt.write(str(wgt_d9d10L2[2])+'\n')
        txt.close()        
        txt=open('d3z2r2pxpxd9d10L2','a')                                  
        txt.write(str(wgt_d9d10L2[3])+'\n')
        txt.close()         
        wgt_d9d10L2[4]=wgt_d9d10L2[0]+wgt_d9d10L2[1]+wgt_d9d10L2[2]+wgt_d9d10L2[3]
        txt=open('d9d10L2','a')                                  
        txt.write(str(wgt_d9d10L2[4])+'\n')
        txt.close()         
        
        
        txt=open('dx2y2pxpxd10d9L2','a')                                  
        txt.write(str(wgt_d10d9L2[0])+'\n')
        txt.close()
        txt=open('dx2y2pxpyd10d9L2','a')                                  
        txt.write(str(wgt_d10d9L2[1])+'\n')
        txt.close()    
        txt=open('d3z2r2pxpxd10d9L2','a')                                  
        txt.write(str(wgt_d10d9L2[2])+'\n')
        txt.close()
        txt=open('d3z2r2pxpyd10d9L2','a')                                  
        txt.write(str(wgt_d10d9L2[3])+'\n')
        txt.close() 
        wgt_d10d9L2[4]=wgt_d10d9L2[0]+wgt_d10d9L2[1]+wgt_d10d9L2[2]+wgt_d10d9L2[3]
        txt=open('d10d9L2','a')                                  
        txt.write(str(wgt_d10d9L2[4])+'\n')
        txt.close() 
        
        
        txt=open('dx2y2pxpxd9L2d10','a')                                  
        txt.write(str(wgt_d9L2d10[0])+'\n')
        txt.close()
        txt=open('dx2y2pxpyd9L2d10','a')                                  
        txt.write(str(wgt_d9L2d10[1])+'\n')
        txt.close()    
        txt=open('d3z2r2pxpxd9L2d10','a')                                  
        txt.write(str(wgt_d9L2d10[2])+'\n')
        txt.close()
        txt=open('d3z2r2pxpyd9L2d10','a')                                  
        txt.write(str(wgt_d9L2d10[3])+'\n')
        txt.close()     
        wgt_d9L2d10[4]=wgt_d9L2d10[0]+wgt_d9L2d10[1]+wgt_d9L2d10[2]+wgt_d9L2d10[3]
        txt=open('d9L2d10','a')                                  
        txt.write(str(wgt_d9L2d10[4])+'\n')
        txt.close()         
        
        
        txt=open('dx2y2pxpxd10Ld9L','a')                                  
        txt.write(str(wgt_d10Ld9L[0])+'\n')
        txt.close()
        txt=open('dx2y2pxpyd10Ld9L','a')                                  
        txt.write(str(wgt_d10Ld9L[1])+'\n')
        txt.close()    
        txt=open('d3z2r2pxpxd10Ld9L','a')                                  
        txt.write(str(wgt_d10Ld9L[2])+'\n')
        txt.close()
        txt=open('d3z2r2pxpyd10Ld9L','a')                                  
        txt.write(str(wgt_d10Ld9L[3])+'\n')
        txt.close()   
        wgt_d10Ld9L[4]=wgt_d10Ld9L[0]+wgt_d10Ld9L[1]+wgt_d10Ld9L[2]+wgt_d10Ld9L[3]
        txt=open('d10Ld9L','a')                                  
        txt.write(str(wgt_d10Ld9L[4])+'\n')
        txt.close()          
        
        
        txt=open('dx2y2pxpxd9Ld10L','a')                                  
        txt.write(str(wgt_d9Ld10L[0])+'\n')
        txt.close()
        txt=open('dx2y2pxpyd9Ld10L','a')                                  
        txt.write(str(wgt_d9Ld10L[1])+'\n')
        txt.close()    
        txt=open('d3z2r2pxpxd9Ld10L','a')                                  
        txt.write(str(wgt_d9Ld10L[2])+'\n')
        txt.close()
        txt=open('d3z2r2pxpyd9Ld10L','a')                                  
        txt.write(str(wgt_d9Ld10L[3])+'\n')
        txt.close() 
        wgt_d9Ld10L[4]=wgt_d9Ld10L[0]+wgt_d9Ld10L[1]+wgt_d9Ld10L[2]+wgt_d9Ld10L[3]
        txt=open('d9Ld10L','a')                                  
        txt.write(str(wgt_d9Ld10L[4])+'\n')
        txt.close()          
        
        
        txt=open('dx2y2pxpxd10L2d9','a')                                  
        txt.write(str(wgt_d10L2d9[0])+'\n')
        txt.close()
        txt=open('dx2y2pxpyd10L2d9','a')                                  
        txt.write(str(wgt_d10L2d9[1])+'\n')
        txt.close()    
        txt=open('d3z2r2pxpxd10L2d9','a')                                  
        txt.write(str(wgt_d10L2d9[2])+'\n')
        txt.close()
        txt=open('d3z2r2pxpyd10L2d9','a')                                  
        txt.write(str(wgt_d10L2d9[3])+'\n')
        txt.close() 
        wgt_d10L2d9[4]=wgt_d10L2d9[0]+wgt_d10L2d9[1]+wgt_d10L2d9[2]+wgt_d10L2d9[3]
        txt=open('d10L2d9','a')                                  
        txt.write(str(wgt_d10L2d9[4])+'\n')
        txt.close()          
        
        
        txt=open('d3z2r2dx2y2pxd10d8LS1','a')                                  
        txt.write(str(wgt_d10d8L[0])+'\n')
        txt.close()
        txt=open('d3z2r2dx2y2pxd10d8L','a')                                  
        txt.write(str(wgt_d10d8L[1])+'\n')
        txt.close()

        
        txt=open('d3z2r2dx2y2pxd9Ld9','a')                                  
        txt.write(str(wgt_d9Ld9[0])+'\n')
        txt.close()  
        txt=open('dx2y2d3z2r2pxd9Ld9','a')                                  
        txt.write(str(wgt_d9Ld9[1])+'\n')
        txt.close() 
        wgt_d9Ld9[4]=wgt_d9Ld9[0]+wgt_d9Ld9[1]
        txt=open('d9Ld9','a')                                  
        txt.write(str(wgt_d9Ld9[4])+'\n')
        txt.close()        
        
        txt=open('d3z2r2dx2y2pxd8d10LS1','a')                                  
        txt.write(str(wgt_d8d10L[0])+'\n')
        txt.close()  
        txt=open('d3z2r2dx2y2pxd8d10L','a')                                  
        txt.write(str(wgt_d8d10L[1])+'\n')
        txt.close()  
        
        txt=open('d3z2r2dx2y2pxd8Ld10S1','a')                                  
        txt.write(str(wgt_d8Ld10[0])+'\n')
        txt.close()  
        txt=open('d3z2r2dx2y2pxd8Ld10','a')                                  
        txt.write(str(wgt_d8Ld10[1])+'\n')
        txt.close()  

                
        txt=open('d3z2r2dx2y2pxd10Ld8S1','a')                                  
        txt.write(str(wgt_d10Ld8[0])+'\n')
        txt.close()  
        txt=open('d3z2r2dx2y2pxd10Ld8','a')                                  
        txt.write(str(wgt_d10Ld8[1])+'\n')
        txt.close() 

        wgt_d10Ld8[2]=wgt_d8d10L[0]+wgt_d8Ld10[0]+wgt_d10Ld8[0]+wgt_d10d8L[0]+wgt_d8d10L[1]+wgt_d8Ld10[1]+wgt_d10Ld8[1]+wgt_d10d8L[1]    
        txt=open('d10d8andd8d10','a')                                  
        txt.write(str(wgt_d10Ld8[2])+'\n')
        txt.close()                  
        
        txt=open('d10d10','a')                                  
        txt.write(str(wgt_d10d10[0])+'\n')
        txt.close()      
        
        sumweight2=wgt_d9d8[4]+wgt_d8d9[4]+wgt_d9d9L[3]+wgt_d9d10L2[4]+wgt_d10d9L2[4]+wgt_d10Ld9L[4]+wgt_d9Ld10L[4]+wgt_d10L2d9[4]
        sumweight2=sumweight2+wgt_d9Ld9[4]+wgt_d10Ld8[2]+wgt_d10d10[0]+ wgt_d9L2d10[4]

        print ('sumweight2=',sumweight2)

        
    print("--- get_ground_state %s seconds ---" % (time.time() - t1))
                
    return vals, vecs 

#########################################################################
    # set up Lanczos solver
#     dim  = VS.dim
#     scratch = np.empty(dim, dtype = complex)
    
#     #`x0`: Starting vector. Use something randomly initialized
#     Phi0 = np.zeros(dim, dtype = complex)
#     Phi0[10] = 1.0
    
#     vecs = np.zeros(dim, dtype = complex)
#     solver = lanczos.LanczosSolver(maxiter = 200, 
#                                    precision = 1e-12, 
#                                    cond = 'UPTOMAX', 
#                                    eps = 1e-8)
#     vals = solver.lanczos(x0=Phi0, scratch=scratch, y=vecs, H=matrix)
#     print ('GS energy = ', vals)
    
#     # get state components in GS; note that indices is a tuple
#     indices = np.nonzero(abs(vecs)>0.01)
#     wgt_d8 = np.zeros(6)
#     wgt_d9L = np.zeros(4)
#     wgt_d10L2 = np.zeros(1)

#     print ("Compute the weights in GS (lowest Aw peak)")
#     #for i in indices[0]:
#     for i in range(0,len(vecs)):
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
 
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         orb1 = state['hole1_orb']
#         orb2 = state['hole2_orb']
#         orb3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']

#         #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
#         #    continue
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         o12 = sorted([orb1,orb2,orb3])
#         o12 = tuple(o12)

#         if i in indices[0]:
#             print (' state ', orb1,s1,x1,y1,z1,orb2,s2,x2,y2,z2,orb3,s3,x3,y3,z3 ,'S=',S12,'Sz=',Sz12,", weight = ", abs(vecs[i,k])**2)
#     return vals, vecs, wgt_d8, wgt_d9L, wgt_d10L2
