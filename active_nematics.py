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

u = dist.VectorField(coord, name='u', bases=(xbasis,ybasis)) #velcoity field
#Q=dist.TensorField(coords,name='Q',bases=(xbasis,ybasis),components=2)# we comment this it will be solved later
#Rembmer Q is symmetric and traceless we define first as scalar fields
Qxx=dist.ScalarField(name='Qxx',bases=(xbasis,ybasis))
Qyy=dist.ScalarField(name='Qyy',bases=(xbasis,ybasis))
Qyy=-Qxx
Qxy=dist.ScalarField(name='Qxy',bases=(xbasis,ybasis))