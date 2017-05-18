from quspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from quspin.basis import fermion_basis_1d # Hilbert space spin basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation tool
import numpy as np # generic math functions
import matplotlib.pyplot as plt
try: # import python 3 zip function in python 2 and pass if using python 3
    import itertools.izip as zip
except ImportError:
    pass 
##### define model parameters #####
L=100 # system size
J=1.0 # uniform hopping contribution
deltaJ=0.1 # bond dimerisation
Delta=0.0 # staggered potential
beta=1.0/100.0 # set inverse temperature for Fermi-Dirac distribution
##### construct single-particle Hamiltonian #####
# define site-coupling lists
hop_pm=[[-J-deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
hop_mp=[[+J+deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
stagg_pot=[[Delta*(-1)**i,i] for i in range(L)]	
# define basis
basis=fermion_basis_1d(L,Nf=1)
basis_args = (L,)
blocks=[dict(Nf=1,kblock=i,a=2) for i in range(L//2)]
# define static and dynamic lists
static=[["+-",hop_pm],["-+",hop_mp],['n',stagg_pot]]
dynamic=[]
# build real-space Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
# diagonalise real-space Hamiltonian
E,V=H.eigh()
# compute Fourier transform and momentum-space Hamiltonian
FT,Hblock = block_diag_hamiltonian(blocks,static,dynamic,fermion_basis_1d,basis_args,np.complex128
									,get_proj_kwargs=dict(pcon=True))
# diagonalise momentum-space Hamiltonian
Eblock,Vblock=Hblock.eigh()
##### prepare the density observables and initial states #####
# construct Fermi sea
psi_0=Vblock
# construct operators $n_{j=0}$
n_i_static=[['n',[[1.0,0]]]]
n_i=hamiltonian(n_i_static,[],basis=basis,dtype=np.float64,check_herm=False,check_pcon=False)
# construct operators $n_{j=L/2}$
n_f_static=[['n',[[1.0,L//2]]]]
n_f=hamiltonian(n_f_static,[],basis=basis,dtype=np.float64,check_herm=False,check_pcon=False)
# transform n_j operators to momentum space
n_i=n_i.rotate_by(FT,generator=False)
n_f=n_f.rotate_by(FT,generator=False)
##### evaluate nonequal time correlator <FS|n_f(t) n_i(0)|FS> #####
# define time vector
t=np.linspace(0.0,90.0,901)
# calcualte state acted an by n_i
psi_n_0=n_i.dot(psi_0)
# construct time-evolution operator using exp_op class (sometimes faster)
U = exp_op(Hblock,a=-1j,start=t.min(),stop=t.max(),num=len(t),iterate=True)
# evolve states
psi_t=U.dot(psi_0)
psi_n_t = U.dot(psi_n_0)
# alternative method for time evolution  using Hamiltonian class
#psi_t=Hblock.evolve(psi_0,0.0,t,iterate=True)
#psi_n_t=Hblock.evolve(psi_n_0,0.0,t,iterate=True)
# compute correlator
n_i_0=n_i.matrix_ele(psi_0,psi_0,diagonal=False) # expectation of n_i at t=0
# preallocate variable
correlators=np.zeros(t.shape+psi_0.shape[1:])
expectations=np.zeros(t.shape+psi_0.shape[1:])
# loop over the time-evolved states
for i, (psi,psi_n) in enumerate( zip(psi_t,psi_n_t) ):
	correlators[i,:]=n_f.matrix_ele(psi,psi_n,diagonal=True).real
	expectations[i,:]=n_f.matrix_ele(psi,psi,diagonal=True)
# evaluate correlator at finite temperature
n_FD=1.0/(np.exp(beta*E)+1.0)
print((n_FD*correlators).sum(axis=-1).shape, (n_FD*expectations).sum(axis=-1).shape, (n_FD*n_i_0).sum().shape)
correlator = (n_FD*correlators).sum(axis=-1) - (n_FD*expectations).sum(axis=-1)*(n_FD*n_i_0).sum()
##### plot data #####
#plt.scatter(np.arange(L),E)
#plt.show()

plt.plot(t,correlator)
plt.show()




