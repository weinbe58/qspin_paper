from __future__ import print_function, division
from quspin.operators import ops_dict,hamiltonian,exp_op
from quspin.basis import spin_basis_1d # spin basis constructor
from quspin.tools.measurements import obs_vs_time # calculating dynamics
from quspin.tools.Floquet import Floquet_t_vec # period-spaced time vector
import numpy as np # general math functions
import matplotlib.pyplot as plt
#
###### define model parameters
L_12 = 18 # length of chain for spin 1/2
L_1 = 11 # length of chain for spin 1
Omega = 2.0 # drive frequency
A = 2.0 # drive amplitude
#
###### setting up bases ######
basis_12 = spin_basis_1d(L_12,S="1/2",kblock=0,pblock=1,zblock=1) # spin 1/2 basis
basis_1 = spin_basis_1d(L_1,S="1"  ,kblock=0,pblock=1,zblock=1) # spin 1 basis
# print information about the basis
print("S = {S:3s}, L = {L:2d}, Size of H-space: {Ns:d}".format(S="1/2",L=L_12,Ns=basis_12.Ns))
print("S = {S:3s}, L = {L:2d}, Size of H-space: {Ns:d}".format(S="1"  ,L=L_1,Ns=basis_1.Ns))
#
###### setting up operators in hamiltonian ######
# setting up site-coupling lists
Jzz_12 = [[-1.0,i,(i+1)%L_12] for i in range(L_12)]
hx_12  = [[-1.0,i] for i in range(L_12)]
Jzz_1 = [[-1.0,i,(i+1)%L_1] for i in range(L_1)]
hx_1  = [[-1.0,i] for i in range(L_1)]
# spin-1/2
Hzz_12 = hamiltonian([["zz",Jzz_12]],[],basis=basis_12,dtype=np.float64)
Hx_12  = hamiltonian([["+",hx_12],["-",hx_12]],[],basis=basis_12,dtype=np.float64)
# spin-1
Hzz_1 = hamiltonian([["zz",Jzz_1]],[],basis=basis_1,dtype=np.float64)
Hx_1  = hamiltonian([["+",hx_1],["-",hx_1]],[],basis=basis_1,dtype=np.float64)
#
###### calculate initial states ######
# calculating bandwidth for non-driven hamiltonian
[E_12_min],psi_12 = Hzz_12.eigsh(k=1,which="SA") #
[E_1_min],psi_1 = Hzz_1.eigsh(k=1,which="SA")
# set up the initial states
psi0_12 = psi_12.ravel()
psi0_1 = psi_1.ravel()
#
###### time evolution ######
# stroboscopic time vector
nT = 200 # number of periods to evolve to
t=Floquet_t_vec(Omega,nT,len_T=1) # t.vals=t, t.i=initial time, t.T=drive period
# creating generators of time evolution using exp_op class
U1_12 = exp_op(Hzz_12+A*Hx_12,a=-1j*t.T/4)
U2_12 = exp_op(Hzz_12-A*Hx_12,a=-1j*t.T/2)
U1_1 = exp_op(Hzz_1+A*Hx_1,a=-1j*t.T/4)
U2_1 = exp_op(Hzz_1-A*Hx_1,a=-1j*t.T/2)
# user-defined generator for stroboscopic dynamics 
def evolve_gen(psi0,nT,*U_list):
	yield psi0
	for i in range(nT): # loop over number of periods
		for U in U_list: # loop over unitaries
			psi0 = U.dot(psi0)
		yield psi0
# get generator objects for time-evolved states
psi_12_t = evolve_gen(psi0_12,nT,U2_12,U1_12,U2_12)
psi_1_t = evolve_gen(psi0_1,nT,U2_1,U1_1,U2_1)
#
###### compute expectation values of observables ######
# measure Hzz as a function of time
Obs_12_t = obs_vs_time(psi_12_t,t.vals,dict(E=Hzz_12),return_state=True)
Obs_1_t = obs_vs_time(psi_1_t,t.vals,dict(E=Hzz_1),return_state=True)
# calculating the entanglement entropy density
Sent_t_12 = basis_12.ent_entropy(Obs_12_t["psi_t"],sub_sys_A=range(L_12//2))["Sent_A"]/(L_12//2)
Sent_t_1 = basis_1.ent_entropy(Obs_1_t["psi_t"],sub_sys_A=range(L_1//2))["Sent_A"]/(L_1//2)
# calculate Page entropy density values
s_p_12 = np.log(2)-2.0**(-L_12//2-L_12)/(2*(L_12//2))
s_p_1 = np.log(3)-3.0**(-L_1//2-L_1)/(2*(L_1//2))
#
###### plotting results ######
plt.plot(t.strobo.inds,(Obs_12_t["E"]-E_12_min)/(-E_12_min),marker='.',markersize=5,label="$S=1/2$")
plt.plot(t.strobo.inds,(Obs_1_t["E"]-E_1_min)/(-E_1_min),marker='.',markersize=5,label="$S=1$")
plt.grid()
plt.ylabel("$Q(t)$",fontsize=20)
plt.xlabel("$t/T$",fontsize=20)
plt.savefig("TFIM_Q.pdf")
plt.figure()
plt.plot(t.strobo.inds,Sent_t_12/s_p_12,marker='.',markersize=5,label="$S=1/2$")
plt.plot(t.strobo.inds,Sent_t_1/s_p_1,marker='.',markersize=5,label="$S=1$")
plt.grid()
plt.ylabel("$s_{\mathrm{ent}}(t)/s_\mathrm{Page}$",fontsize=20)
plt.xlabel("$t/T$",fontsize=20)
plt.legend(loc=0,fontsize=16)
plt.savefig("TFIM_S.pdf")
plt.show()