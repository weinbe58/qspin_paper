from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d, fermion_basis_1d # Hilbert space spin basis
import numpy as np

L=8
basis_fermion = fermion_basis_1d(L=L,Nf=range(0,L+1,2),kblock=0)
J_pm=[[-1,i,i+1] for i in range(L-1)] 
J_mp=[[+1,i,i+1] for i in range(L-1)]
J_pm.append([+1,L-1,0]) # APBC
J_mp.append([-1,L-1,0]) # APBC
static_fermion =[["+-",J_pm],["-+",J_mp]]
H_fermion=hamiltonian(static_fermion,[],basis=basis_fermion,dtype=np.float64,check_pcon=False)