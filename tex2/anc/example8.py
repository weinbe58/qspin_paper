from __future__ import print_function, division

from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from quspin.tools.block_tools import block_ops
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

""" schematic of how the ladder lattic is set up

coupling parameters:
-: J_par_1
^: J_par_2
|: J_perp

^1^3^5^7^9^
 | | | | |
-0-2-4-6-8-

translations (i -> i+2):

^9^1^3^5^7^
 | | | | | 
-8-0-2-4-6-


if J_par_1 same as J_par_2 then one can use parity

parity (i -> N - i):


-8-6-4-2-0-
 | | | | |
-9-7-5-3-1-

"""
np.random.seed(0)

L = 6
N = 2*L
nb = 0.5
sps = 2


J_par_1 = 1.0
J_par_2 = 1.0
J_perp =  1.0
U = 10.0



basis = boson_basis_1d(N,nb=nb,sps=sps)

int_list_2 = [[U,i,i] for i in range(N)]
int_list_1 = [[-U,i] for i in range(N)]
hop_list = [[-J_par_1,i,(i+2)%N] for i in range(0,N,2)]
hop_list.extend([[-J_par_2,i,(i+2)%N] for i in range(1,N,2)])
hop_list.extend([[-J_perp,i,i+1] for i in range(0,N,2)])
hop_list_hc = [[J.conjugate(),i,j] for J,i,j in hop_list]

static = [
			["+-",hop_list],
			["-+",hop_list_hc],
			["nn",int_list_2],
			["n",int_list_1]
		]
dynamic = []

no_checks = dict(check_herm=False,check_symm=False,check_pcon=False)

i0 = np.random.randint(basis.Ns) # pick random state from basis set
psi = np.zeros(basis.Ns,dtype=np.float64)
psi[i0] = 1.0
# print info about setup
state_str = "".join(str(int((basis[i0]//basis.sps**i)%basis.sps)) for i in range(N))
print("H-space size: {}, initial state: |{}>".format(basis.Ns,state_str))
blocks=[]


if J_par_1 == J_par_2:
	for kblock in range(L//2+1):
		for pblock in [-1,1]:
			blocks.append(dict(nb=nb,sps=sps,a=2,kblock=kblock,pblock=pblock))
else:
	for kblock in range(L):
		blocks.append(dict(nb=nb,sps=sps,a=2,kblock=kblock))




U_block = block_ops(blocks,static,dynamic,boson_basis_1d,(N,),np.complex128,get_proj_kwargs=dict(pcon=True))

start,stop,num = 0,15,151
psi_t = U_block.expm(psi,start=start,stop=stop,num=num,block_diag=False)


# plotting movie
state_str = np.array([int(s) for s in state_str])
ind_start = np.argwhere(state_str>0).ravel()
sub_sys_A = range(0,N,2) # half of chain is subsystem 
# observables
n_list = [hamiltonian([["n",[[1.0,i]]]],[],basis=basis,dtype=np.float64,**no_checks) for i in range(N)]
times = np.linspace(start,stop,num)


ent_t = np.fromiter((basis.ent_entropy(psi,sub_sys_A=sub_sys_A)["Sent_A"]/L for psi in psi_t.T[:]),dtype=np.float64,count=num)
expt_n_t = np.vstack([n.expt_value(psi_t).real for n in n_list]).T

n_t = np.zeros((num,2,L))
n_t[:,0,:] = expt_n_t[:,0::2]
n_t[:,1,:] = expt_n_t[:,1::2]


# plotting static figures

fig, ax = plt.subplots(nrows=5,ncols=1)

im=[]
im_ind = []
for i,t in enumerate(np.logspace(-1,np.log10(stop-1),5,base=10)):
	j = times.searchsorted(t)
	im_ind.append(j)
	im.append(ax[i].imshow(n_t[j],cmap="hot",vmax=n_t.max(),vmin=0))
	ax[i].tick_params(labelbottom=False,labelleft=False)

cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
fig.colorbar(im[2],cax)
plt.savefig("boson_density.pdf")
plt.figure()
plt.plot(times,ent_t,lw=2)
plt.plot(times[im_ind],ent_t[im_ind],marker="o",linestyle="",color="red")
plt.xlabel("$t/J$",fontsize=20)
plt.ylabel("$s_\mathrm{ent}(t)$",fontsize=20)
plt.grid()
plt.savefig("boson_entropy.pdf")
plt.show()


# setting up two plots to animate side by side
"""
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(10, 5)

ax1.set_xlabel(r"$t/J$",fontsize=18)
ax1.set_ylabel(r"$s_\mathrm{ent}$",fontsize=18)

ax1.grid()
line1, = ax1.plot(times, ent_t, lw=2)
line1.set_data([],[])


im = ax2.matshow(n_t[0],cmap="hot")
fig.colorbar(im)


def run(i): # function to update frame
	# set new data for plots
	line1.set_data(times[:i],ent_t[:i])
	im.set_data(n_t[i])

	return im, line1

ani = animation.FuncAnimation(fig, run, range(num),interval=50)
plt.show()
"""