from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d
from exact_diag_py.tools import observables
from exact_diag_py.tools.Floquet import Floquet

import numpy as np

##### define model parameters #####
L=4 # system size

J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field

# define operators with PBC
x_field=[[1.0,i] for i in range(L)]
J_nn=[[1.0,i,(i+1)%L] for i in range(L)]
J_nnn=[[1.0,i,(i+1)%L,(i+2)%L] for i in range(L)]

basis=spin_basis_1d(L=L,kblock=0,pblock=1)

# define operators
x = hamiltonian([['x',x_field]],[],dtype=np.float64,basis=basis)
y = hamiltonian([['y',x_field]],[],dtype=np.complex128,basis=basis)
z = hamiltonian([['z',x_field]],[],dtype=np.float64,basis=basis)

zz = hamiltonian([['zz',J_nn]],[],dtype=np.float64,basis=basis)
yy = hamiltonian([['yy',J_nn]],[],dtype=np.float64,basis=basis)
xx = hamiltonian([['xx',J_nn]],[],dtype=np.float64,basis=basis)

zy_yz = hamiltonian([['zy',J_nn],['yz',J_nn]],[],dtype=np.complex128,basis=basis)
zx_xz = hamiltonian([['zx',J_nn],['xz',J_nn]],[],dtype=np.float64,basis=basis)

zxz = hamiltonian([['zxz',J_nnn]],[],dtype=np.float64,basis=basis)

# build building blocks
H0 		 = 0.5*(J*zz + h*z + g*x)
tilde_H0 = 0.5*(J*zz + h*z - g*x)

Keff_2 = g*(J*zy_yz + h*y)

### check commutators
comm1 = tilde_H0.comm(H0.tocsr() )

#print np.linalg.norm( comm1 - -1j*Keff_2.todense() )

comm2 = -tilde_H0.comm(comm1)

comm3 = -tilde_H0.comm(-1j*Keff_2.tocsr())

#print np.linalg.norm( comm2.todense() - comm3.todense() )



# check commutators one by one

A1 = J*zy_yz
A2 = h*y
B1 = J*zz
B2 = -g*x
B3 = h*z

'''
print np.linalg.norm( B1.comm(A1.tocsr()) - 4.0j*J**2*(zxz + x).todense() )
print np.linalg.norm( B2.comm(A1.tocsr()) - -4.0j*J*g*(yy-zz).todense() )
print np.linalg.norm( B3.comm(A1.tocsr()) - 2.0j*J*h*(zx_xz).todense() )
print np.linalg.norm( B1.comm(A2.tocsr()) - 2.0j*J*h*(zx_xz).todense() )
print np.linalg.norm( B2.comm(A2.tocsr()) - 2.0j*g*h*z.todense() )
print np.linalg.norm( B3.comm(A2.tocsr()) - 2.0j*h**2*x.todense() )
'''

#print np.linalg.norm( B3.comm(A1.tocsr()).todense() - B1.comm(A2.tocsr()).todense() )


Heff_2 =  -(2.0*J**2*g*(zxz + x) + 2.0*J*h*g*(zx_xz) - 2.0*J*g**2*(yy - zz) + g*h**2*x + h*g**2*z )


comm4 = g/2.0*1j*( B1.comm(A1.tocsr()) + B2.comm(A1.tocsr()) + B3.comm(A1.tocsr()) + B1.comm(A2.tocsr()) + B2.comm(A2.tocsr()) + B3.comm(A2.tocsr())  )

comm5 = g/2.0*( (B1+B2+B3).comm( 1j*(A1+A2).tocsr() ) )

# J=0
#print np.linalg.norm( -g/2.0*( 2.0*g*h*z + 2.0*h**2*x ).todense() - comm4.todense() )

# h=0
#print np.linalg.norm( -(2.0*J**2*g*(zxz + x) - 2.0*J*g**2*(yy - zz) ).todense() - comm4.todense() )

#print np.linalg.norm( comm4.todense() - comm5.todense() )

#print np.linalg.norm( 1j*Keff_2.todense() - 1j*(A1+A2).todense() )

#print np.linalg.norm( tilde_H0.todense() - 0.5*(B1+B2+B3).todense() )

#print np.linalg.norm( comm4.todense() - Heff_2.todense() )

#print np.linalg.norm( comm5.todense() - comm2.todense() )

#exit()

#print np.linalg.norm( comm5.todense() - comm2 )

#exit()


#print Heff_2
print comm1


print np.linalg.norm( comm2 - Heff_2.todense() )




