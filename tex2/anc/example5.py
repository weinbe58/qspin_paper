from __future__ import print_function, division

import sys,os
import argparse

qspin_path = os.path.join(os.getcwd(),"../../../QuSpin_dev/")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import fermion_basis_1d # Hilbert space spin basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation tool
import numpy as np # generic math functions
import matplotlib.pyplot as plt
try: # import python 3 zip function in python 2 and pass if python 3
    import itertools.izip as zip
except ImportError:
    pass 
	


# define model params
L=10 # system size
J=1.0 #uniform hopping
deltaJ=0.2 # hopping difference
Delta=0.5 # staggered potential
# define site-coupling lists
hop_pm=[[+J+deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
hop_mp=[[-J-deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
stagg_pot=[[Delta*(-1)**i,i] for i in range(L)]	
# define basis
basis=fermion_basis_1d(L,Nf=1)
basis_args = (L,)
blocks=[dict(Nf=1,kblock=i,a=2) for i in range(L//2)]
# define static and dynamic lists
static=[["+-",hop_pm],["-+",hop_mp],['n',stagg_pot]]
dynamic=[]
#### calculate Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
E,V=H.eigh()
# define Fourier transform and momentum-space Hamiltonian
FT,Hblock = block_diag_hamiltonian(blocks,static,dynamic,fermion_basis_1d,basis_args,np.complex128,get_proj_kwargs=dict(pcon=True))
Eblock=Hblock.eigvalsh()


#plt.scatter(np.arange(L),E)
#plt.show()

# construct Fermi sea
psi_FS=V[:,5]

# construct operator n_{j=0}
n_static=[['n',[[1.0,0]]]]
n_j=hamiltonian(n_static,[],basis=basis,dtype=np.float64,check_herm=False,check_pcon=False)
# transform n_j to momentum space
n_k=n_j.rotate_by(FT,generator=False)
# evaluate nonequal time correlator <FS|n_j(t) n_j(0)|FS>
t=np.linspace(0.0,30.0,300)
npsi=n_k.dot(psi_FS)
psi_t=Hblock.evolve(psi_FS,0.0,t,iterate=True)
npsi_t=Hblock.evolve( npsi,0.0,t,iterate=True)
# compute correlator
n_k_0=n_k.matrix_ele(psi_FS,psi_FS,diagonal=True) # expectation of n_k at t=0
correlator=np.zeros(t.shape)
for i, (psi,npsi) in enumerate( zip(psi_t,npsi_t) ):
	correlator[i]=np.sum(  n_k.matrix_ele(psi,npsi,diagonal=True) \
						 - n_k.matrix_ele(psi, psi,diagonal=True)*n_k_0  ).real

plt.plot(t,correlator)
plt.show()






