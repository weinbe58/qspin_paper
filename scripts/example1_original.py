from qspin.operators import hamiltonian
from qspin.basis import spin_basis_1d
from qspin.tools.measurements import ent_entropy, diag_ensemble
from numpy.random import ranf,seed
import numpy as np
from time import time

seed(0) # reset random number generator

##### model parameters ######
L = 10 #system size
n_real = 1 # number of disorder realisations
vs = np.logspace(-2,0,num=100,base=10) # vector of ramp speeds

##### compute spin basis #####
basis = spin_basis_1d(L,Nup=L/2,pauli=False)

# define linear ramp function
def ramp(t,v):
		return 0.5 + v*t
	
def realization(n_real,vs,basis):
	"""
	This function cimputes the physical quantities of interest for a single disorder realisation.
	--- arguments ---
	n_real: number of disorder realisations
	vs: vector of ramp speeds
	basis: spin_basis_1d object containing the spin basis
	"""

	ti = time() # start timer

	L = basis.L # read off the system size

	# model parameters
	h_MBL=3.9 # disorder strength for static system to be MBL
	h_ETH=0.1 # disorder strength for static system to be delocalised
	# draw a random x-field distribution in [-1.0,1.0] for each lattice site
	unscaled_fields = -1+2*ranf((L,))
	print unscaled_fields
	# define operator lists #####
	h_MBL = [[h_MBL*unscaled_fields[i],i] for i in range(L)]
	h_ETH = [[h_ETH*unscaled_fields[i],i] for i in range(L)]
	J_zz = [[1.0,i,i+1] for i in range(L-1)]
	J_xy = [[0.5,i,i+1] for i in range(L-1)]
	# define static and dynamic operators #####
	static_MBL = [["z",h_MBL],["+-",J_xy],["-+",J_xy]]
	static_ETH = [["z",h_ETH],["+-",J_xy],["-+",J_xy]]
	drive_list = [["zz",J_zz]]
	# compute the MBL and ETH Hamiltonians
	H_MBL = hamiltonian(static_MBL,[],basis=basis,dtype=np.float64,check_pcon=False,check_symm=False,check_herm=False)
	H_ETH = hamiltonian(static_ETH,[],basis=basis,dtype=np.float64,check_pcon=False,check_symm=False,check_herm=False)
	# 
	Hd_csr = hamiltonian(drive_list,[],basis=basis,dtype=np.float64,check_pcon=False,check_symm=False,check_herm=False)
	Hd_csr = Hd_csr.tocsr()
	# MBL ramps

	# calculate the energy at infinite temperature for initial hamiltonian
	Emin,Emax = (H_MBL + 0.5*Hd_csr).eigsh(k=2,which="BE",maxiter=1E10,return_eigenvectors=False)
	E_inf_temp = (Emax + Emin)/2.0
	# calculate nearest eigenstate to energy at infinite temperature
	E,psi_0 = (H_MBL + 0.5*Hd_csr).eigsh(k=1,sigma=E_inf_temp,maxiter=1E10)
	psi_0 = psi_0.reshape((-1,))
	# calculate the eigen-basis of the final hamiltonian
	E_final,V_final = (H_MBL + Hd_csr).eigh()
	# evolve states and calculate observables
	run_MBL = [do_ramp(basis,H_MBL,Hd_csr,v,psi_0,E_final,V_final) for v in vs]
	run_MBL = np.vstack(run_MBL).T

	
	# ETH ramps

	# calculate the energy at infinite temperature for initial hamiltonian
	Emin,Emax = (H_ETH + 0.5*Hd_csr).eigsh(k=2,which="BE",maxiter=1E10,return_eigenvectors=False)
	E_inf_temp = (Emax + Emin)/2.0
	# calculate nearest eigenstate to energy at infinite temperature for initial hamiltonian
	E,psi_0 = (H_ETH + 0.5*Hd_csr).eigsh(k=1,sigma=E_inf_temp,maxiter=1E10)
	psi_0 = psi_0.reshape((-1,))
	# calculate the eigen-basis of the final hamiltonian
	E_final,V_final = (H_ETH + Hd_csr).eigh()
	# evolve states and calculate observables
	run_ETH = [do_ramp(basis,H_ETH,Hd_csr,v,psi_0,E_final,V_final) for v in vs]
	run_ETH = np.vstack(run_ETH).T

	print "realization {0} finished in {1:.2f} sec".format(n_real,time()-ti)
	return run_MBL,run_ETH


def do_ramp(basis,H,Hd_csr,v,psi_0,E_final,V_final):
	T = 0.5/v # total ramp time
	# define H(t)
	dynamic = [[Hd_csr,ramp,[v]],] # driven zz term
	H = H + hamiltonian([],dynamic,dtype=Hd_csr.dtype)
	# time-evolve Hamiltonian
	psi = H.evolve(psi_0,0.0,T)
	# calculate entanglement entropy
	subsys = range(basis.L/2) # define subsystem
	Sent = ent_entropy(psi,basis,chain_subsys=subsys)['Sent']
	# calculate diagonal entropy in the post ramp basis
	S_d = diag_ensemble(basis.L,psi,E_final,V_final,Sd_Renyi=True)["Sd_pure"]

	return np.array([S_d,Sent])


# read off data
data = np.asarray([realization(i,vs,basis) for i in xrange(n_real)])
run_MBL,run_ETH = zip(*data)
# average over disorder
mean_MBL = np.mean(run_MBL,axis=0)
mean_ETH = np.mean(run_ETH,axis=0)

####### plotting data #########
import matplotlib.pyplot as plt
# first plot
f, pltarr1 = plt.subplots(2,sharex=True)
pltarr1[0].set_title("MBL phase")
pltarr1[0].plot(vs,mean_MBL[0],label="MBL",marker=".",color="blue")
pltarr1[0].set_ylabel("Diagonal Entropy")
pltarr1[0].set_xlabel("Velocity")
pltarr1[0].set_xscale("log")
pltarr1[0].grid(True,which='both')


pltarr1[1].plot(vs,mean_MBL[1],marker=".",color="blue")
pltarr1[1].set_ylabel("Entanglement Entropy")
pltarr1[1].set_xlabel("Velocity")
pltarr1[1].set_xscale("log")
pltarr1[1].grid(True,which='both')


# second plot
f, pltarr2 = plt.subplots(2,sharex=True)
pltarr2[0].set_title("ETH phase")
pltarr2[0].plot(vs,mean_ETH[0],marker=".",color="green")
pltarr2[0].set_ylabel("Diagonal Entropy")
pltarr2[0].set_xlabel("Velocity")
pltarr2[0].set_xscale("log")
pltarr2[0].grid(True,which='both')

pltarr2[1].plot(vs,mean_ETH[1],marker=".",color="green")
pltarr2[1].set_ylabel("Entanglement Entropy")
pltarr2[1].set_xlabel("Velocity")
pltarr2[1].set_xscale("log")
pltarr2[1].grid(True,which='both')

# show plots
plt.show()




