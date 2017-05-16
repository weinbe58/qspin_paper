from __future__ import print_function, division
import sys,os
import numpy as np
from numpy.random import uniform,choice
from quspin.basis import tensor_basis,fermion_basis_1d
from quspin.operators import hamiltonian,exp_op,ops_dict
from quspin.tools.measurements import obs_vs_time
from joblib import Parallel,delayed
import matplotlib.pyplot as plt
from time import time

# setting parameters for simulation
n_jobs = 4 # number of cores to use in calculating realizations
n_real = 100 # number of realizations
n_boot = 100*n_real # number of bootstrap realizations to calculate error
# physical parameters
L = 8 # system size
N = L//2 # number of particles
w1 = 1.0 # disorder strength
w2 = 4.0
w3 = 10.0 
J = 1.0 # hopping strength
U = 5.0 # interaction strength
k = 0.1 # trap stiffness
# range to evolve system
start=0.0
stop=35.0
num=101
# setting up basis
N_up = N//2 + N % 2 # number of fermions with spin up
N_down = N//2 # number of fermions with spin down
# building the two basis to tensor together
basis_up = fermion_basis_1d(L,Nf=N_up) # up basis
basis_down = fermion_basis_1d(L,Nf=N_down) # down basis
basis = tensor_basis(basis_up,basis_down) # spinful fermsions
# creating coupling lists
i_mid = (L//2+1 if L%2 else L//2+0.5) # mid point on lattice
hop_right = [[J,i,i+1] for i in range(L-1)] # hopping to the right
hop_left = [[-J,i,i+1] for i in range(L-1)] # hopping to the left
int_list = [[U,i,i] for i in range(L)] # onsite interaction
trap_list = [[0.5*k*(i-i_mid)**2,i] for i in range(L)] # harmonic trap
# coupling list to create the sublattice imbalance observable
sublat_list = [[(-1)**i,i] for i in range(0,L)]
# create static lists
operator_list_0 = [	
			["+-|", hop_left], # up hop left
			["-+|", hop_right], # up hop right
			["n|", trap_list], # up trap potential
			["|+-", hop_left], # down hop left
			["|-+", hop_right], # down hop right
			["|n", trap_list], # down trap potential
			["n|n", int_list], # onsite interaction
		 ]
# create operator dictionary for ops_dict class
# creates a dictioanry with keys h0,h1,h2,...,hL for local potential
operator_dict = {"h"+str(i):[["n|",[[1.0,i]]],["|n",[[1.0,i]]]] for i in range(L)}
operator_dict["H0"]=operator_list_0
# set up hamiltonian dictionary and observable
no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
H_dict = ops_dict(operator_dict,basis=basis,**no_checks)
I = hamiltonian([["n|",sublat_list],["|n",sublat_list]],[],basis=basis,**no_checks)/N
# strings which represent the initial state
s_up = "".join("1000" for i in range(N_up))
s_down = "".join("0010" for i in range(N_down))
print("H-space size: {:d}, initial state: |{:s}>(x)|{:s}>".format(basis.Ns,s_up,s_down))
# basis.index accepts strings and returns the index 
# which corresponds to that state in the basis list
i_0 = basis.index(s_up,s_down)
psi_0 = np.zeros(basis.Ns)
psi_0[i_0] = 1.0
# set up times
times = np.linspace(start,stop,num=num,endpoint=True)
# define function to do dynamics for different disorder realizations.
def realization(H_dict,I,psi_0,disorder,start,stop,num,i):
	ti = time() # start timing function
	# create a parameter list which specifies the onsite potential with disorder
	parameters = {"h"+str(i):h for i,h in enumerate(disorder)}
	# using the parameters dictionary construct a hamiltonian object with those
	# parameters defined in the list
	H = H_dict.tohamiltonian(parameters)
	# use exp_op to get the evolution operator
	U = exp_op(H,a=-1j,start=start,stop=stop,num=num,endpoint=True,iterate=True)
	psi_t = U.dot(psi_0) # get generator of time evolution
	# use obs_vs_time to evaluate the dynamics
	obs_t = obs_vs_time(psi_t,U.grid,dict(I=I))
	# print reporting the computation time for realization
	print("realization {}/{} completed in {:.2f} s".format(i+1,n_real,time()-ti))
	# return observable values.
	return obs_t["I"]

# machinery for doing parallel realizations loop
I_data_1 = np.vstack(Parallel(n_jobs=n_jobs)(delayed(realization)(H_dict,I,psi_0,uniform(-w1,w1,size=L),start,stop,num,i) for i in range(n_real)))
I_data_2 = np.vstack(Parallel(n_jobs=n_jobs)(delayed(realization)(H_dict,I,psi_0,uniform(-w2,w2,size=L),start,stop,num,i) for i in range(n_real)))
I_data_3 = np.vstack(Parallel(n_jobs=n_jobs)(delayed(realization)(H_dict,I,psi_0,uniform(-w3,w3,size=L),start,stop,num,i) for i in range(n_real)))
# calculating mean and error via bootstrap sampling
I_1 = I_data_1.mean(axis=0) # get mean value of I
I_2 = I_data_2.mean(axis=0) 
I_3 = I_data_3.mean(axis=0) 
# generate bootstrap samples
bootstrap_gen_1 = (I_data_1[choice(n_real,n_real)].mean(axis=0) for i in range(n_boot)) 
bootstrap_gen_2 = (I_data_2[choice(n_real,n_real)].mean(axis=0) for i in range(n_boot)) 
bootstrap_gen_3 = (I_data_3[choice(n_real,n_real)].mean(axis=0) for i in range(n_boot)) 
# generate the fluctuations about the mean of I
sq_fluc_gen_1 = ((bootstrap-I_1)**2 for bootstrap in bootstrap_gen_1) 
sq_fluc_gen_2 = ((bootstrap-I_2)**2 for bootstrap in bootstrap_gen_2)
sq_fluc_gen_3 = ((bootstrap-I_3)**2 for bootstrap in bootstrap_gen_3) 
# error is calculated as the squareroot of mean fluctuations
dI_1 = np.sqrt(sum(sq_fluc_gen_1)/n_boot) 
dI_2 = np.sqrt(sum(sq_fluc_gen_2)/n_boot) 
dI_3 = np.sqrt(sum(sq_fluc_gen_3)/n_boot) 

# plot imbalance with error bars
fig = plt.figure()
plt.xlabel("$t/J$",fontsize=18)
plt.ylabel("$\mathcal{I}$",fontsize=18)
plt.grid(True)
plt.tick_params(labelsize=16)
plt.errorbar(times,I_1,dI_1,marker=".",label="w={:.2f}".format(w1))
plt.errorbar(times,I_2,dI_2,marker=".",label="w={:.2f}".format(w2))
plt.errorbar(times,I_3,dI_3,marker=".",label="w={:.2f}".format(w3))
plt.legend(loc=0)
fig.savefig('fermion_MBL.pdf', bbox_inches='tight')
plt.show()
