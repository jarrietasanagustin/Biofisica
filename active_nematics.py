""" This program calculates the evlulation of a nematic field 
using the diluted limit

Jorge Arrieta April 2026
 """
import numpy as np 
import matplotlib.pyplot as plt
import dedalus.public as d3

Lx=1.0
Ly=1.0
Nx=128
Ny=128

#We define the coordinates here
#We use complex fourier for the basis since 
#we are using periodic boundary conditions
coord=d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coord, dtype=np.complex128)
dealias = 3/2
xbasis = d3.ComplexFourier(coord['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.ComplexFourier(coord['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

#We now define the fields tha we need to calculate. The velocity field and the orientation field that his a tensor

u = dist.ScalarField( name='u', bases=(xbasis,ybasis)) #x component of the velcoity field
v = dist.ScalarField( name='v', bases=(xbasis,ybasis)) #y component of the velcoity field
p = dist.ScalarField( name='p', bases=(xbasis,ybasis)) #pressure field
#Q=dist.TensorField(coords,name='Q',bases=(xbasis,ybasis),components=2)# we comment this it will be solved later

#Rembmer Q is symmetric and traceless we define first as scalar fields
Qxx=dist.ScalarField(name='Qxx',bases=(xbasis,ybasis))
Qyy=dist.ScalarField(name='Qyy',bases=(xbasis,ybasis))
Qyy=-Qxx
Qxy=dist.ScalarField(name='Qxy',bases=(xbasis,ybasis))


#we define the derivatives of the components of the velocity to assemble the strain tensor 
ux=d3.Differentiate(u,coord['x'])
uy=d3.Differentiate(u,coord['y'])
vx=d3.Differentiate(v,coord['x'])
vy=d3.Differentiate(v,coord['y'])
#Strain rate tensor
Exx=ux
Exy=1/2*(uy+vx)
Eyy=vy
#vorticity
Omega=1/2*(uy-vx)
Omegaxx=0
Omegayy=0
Omegaxy=Omega
Omegayx=-Omega

lambda_n=1#tumbling parameter

#A=lambda*E+Omega
A11=lambda_n*Exx+Omegaxx
A12=lambda_n*Exy+Omegaxy
A21=lambda_n*Exy+Omegayx
A22=lambda_n*Eyy+Omegayy
#B=Q+I/3
B11=Qxx+1/3
B12=Qxy
B21=Qxy
B22=Qyy+1/3

#C=lambda*E-Omega
C11=lambda_n*Exx-Omegaxx
C12=lambda_n*Exy-Omegaxy
C21=lambda_n*Exy-Omegayx
C22=lambda_n*Eyy-Omegayy
#D=Q:nabla u
D=Qxx*(ux-vy)+Qyy*(uy+vx)
####################################################################
#We now write explicit expression for S11 and S12
####################################################################

S11=(A11*B11+A12*B21)+(B11*C11+B12*C21)-2*lambda_n*B11*D
S12=A11*B12+A12*B22+(B11*C21+B12*C22)-2*lambda_n*B12*D
trace = Qxx + (-Qxx) #to check that the trace is zero

problem = d3.IVP([u,v,p,Qxx,Qxy], namespace=locals())
#equation for the components of the nematic tensor field
problem.add_equation("dt(Qxx)+u*dx(Qxx)+v*dy(Qxx) -S11= dx(dx(Qxx))+dy(dy(Qxx))")
problem.add_equation("dt(Qxy)+u*dx(Qxy)+v*dy(Qxy) -S12= dx(dx(Qxy))+dy(dy(Qxy))")
#continuity equation for the velocity field
problem.add_equation("dx(u)+dy(v)=0")
#we need to add the momentum equation for the velocity field
problem.add_equation("dx(p)= dx(dx(u))+dy(dy(u))+dx(Qxx)+dy(Qxy)")
problem.add_equation("dy(p)= dx(dx(v))+dy(dy(v))+dx(Qxy)-dy(Qxx)")