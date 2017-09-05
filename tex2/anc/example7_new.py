from __future__ import print_function, division
import numpy as np
from quspin.operators import ops_dict,hamiltonian,exp_op
from quspin.basis import spin_basis_1d,spin_basis_general
from quspin.tools.measurements import obs_vs_time
import matplotlib.pyplot as plt
import sys,os
# user defined generator
# generates stroboscopic dynamics 
def evolve_gen(psi0,nT,*U_list):
	yield psi0
	for i in range(nT): # loop over number of periods
		for U in U_list: # loop over unitaries
			psi0 = U.dot(psi0)
		yield psi0
# frequency and period for driving.
omega = 10
T = 2*np.pi/omega 
nT = 100 # number of periods to evolve to.
times = np.arange(0,nT+1,1)*T
L_1d = 16 # length of chain for spin 1/2
Lx = 4
Ly = 4
N_2d = Lx*Ly # length of chain for spin 1
###### setting up transformations ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx+Lx*y # translation along x-direction
T_y = x+Lx*((y+1)%Ly) # translation along y-direction
P_y = (Lx-x-1)+Lx*y # reflections about y-axis
P_x = x+Lx*(Ly-y-1) # reflections about x-axis
Z   = -(s+1) # spin inversion
###### setting up basis ######
basis_1d = spin_basis_1d(L_1d,kblock=0,pblock=1,zblock=1) # 1d - basis
basis_2d = spin_basis_general(N_2d,kxblock=(T_x,0),kyblock=(T_y,0),
							pxblock=(P_x,0),pyblock=(P_y,0),zblock=(Z,0)) # 2d - basis
# print information about the basis
print("Size of 1D H-space: {Ns:d}".format(Ns=basis_1d.Ns))
print("Size of 2D H-space: {Ns:d}".format(Ns=basis_2d.Ns))
# setting up coupling lists
Jzz_1d = [[-1.0,i,(i+1)%L_1d] for i in range(L_1d)]
hx_1d  = [[-1.0,i] for i in range(L_1d)]
Jzz_2d = [[-1.0,i,T_x[i]] for i in range(N_2d)]
Jzz_2d.extend([[-1.0,i,T_y[i]] for i in range(N_2d)])
hx_2d  = [[-1.0,i] for i in range(N_2d)]
# dictioanry to turn off checks
no_checks = dict(check_symm=False,check_herm=False)
# setting up hamiltonians
Hzz_1d = hamiltonian([["zz",Jzz_1d]],[],basis=basis_1d,dtype=np.float64)
Hx_1d  = hamiltonian([["x",hx_1d]],[],basis=basis_1d,dtype=np.float64)
Hzz_2d = hamiltonian([["zz",Jzz_2d]],[],basis=basis_2d,dtype=np.float64,**no_checks)
Hx_2d  = hamiltonian([["x",hx_2d]],[],basis=basis_2d,dtype=np.float64,**no_checks)
# calculating bandwidth for non-driven hamiltonian
[E_1d_min],psi_1d = Hzz_1d.eigsh(k=1,which="SA")
[E_2d_min],psi_2d = Hzz_2d.eigsh(k=1,which="SA")
# setting up initial states
psi0_1d = psi_1d.ravel()
psi0_2d = psi_2d.ravel()
# creating generators of time evolution
U1_1d = exp_op(Hzz_1d+omega*Hx_1d,a=-1j*T/4)
U2_1d = exp_op(Hzz_1d-omega*Hx_1d,a=-1j*T/2)
U1_2d = exp_op(Hzz_2d+omega*Hx_2d,a=-1j*T/4)
U2_2d = exp_op(Hzz_2d-omega*Hx_2d,a=-1j*T/2)
# get generator objects to get time dependent states
psi_1d_t = evolve_gen(psi0_1d,nT,U1_1d,U2_1d,U1_1d)
psi_2d_t = evolve_gen(psi0_2d,nT,U1_2d,U2_2d,U1_2d)
# measure energy as a function of time
Obs_1d_t = obs_vs_time(psi_1d_t,times,dict(E=Hzz_1d),return_state=True)
Obs_2d_t = obs_vs_time(psi_2d_t,times,dict(E=Hzz_2d),return_state=True)
# calculate page entropy density
s_p_1d = np.log(2)-2.0**(-L_1d//2-L_1d)/(2*(L_1d//2))
s_p_2d = np.log(2)-2.0**(-N_2d//2-N_2d)/(2*(N_2d//2))
# calculating the entanglement entropy density
Sent_time_1d = basis_1d.ent_entropy(Obs_1d_t["psi_t"],sub_sys_A=range(L_1d//2))["Sent_A"]/(L_1d//2)
Sent_time_2d = basis_2d.ent_entropy(Obs_2d_t["psi_t"],sub_sys_A=range(N_2d//2))["Sent_A"]/(N_2d//2)
#plotting results
plt.plot(times/T,(Obs_1d_t["E"]-E_1d_min)/(-E_1d_min),marker='.',markersize=5,label="$S=1/2$")
plt.plot(times/T,(Obs_2d_t["E"]-E_2d_min)/(-E_2d_min),marker='.',markersize=5,label="$S=1$")
plt.grid()
plt.ylabel("$Q(t)$",fontsize=20)
plt.xlabel("$t/T$",fontsize=20)
plt.savefig("TFIM_Q.pdf")
plt.figure()
plt.plot(times/T,Sent_time_1d/s_p_1d,marker='.',markersize=5,label="$1D$")
plt.plot(times/T,Sent_time_2d/s_p_2d,marker='.',markersize=5,label="$2D$")
plt.grid()
plt.ylabel("$s_{\mathrm{ent}}(t)/s_\mathrm{Page}$",fontsize=20)
plt.xlabel("$t/T$",fontsize=20)
plt.legend(loc=0,fontsize=16)
plt.savefig("TFIM_S.pdf")
plt.show()