""" This program calculates the evlulation of a nematic field 
using the diluted limit

Jorge Arrieta April 2026
 """
import numpy as np 
import matplotlib.pyplot as plt
import dedalus.public as d3

Lx=1.0
Ly=1.0
Nx=256
Ny=256
Pe = 1000e0 #Peclet numberr
lambda_n = 1.0 # tumbling parameter
#We define the coordinates here
#We use complex fourier for the basis since 
#we are using periodic boundary conditions
coord=d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coord, dtype=np.complex128)
dealias = 3/2
xbasis = d3.ComplexFourier(coord['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.ComplexFourier(coord['y'],     size=Ny, bounds=(0, Ly), dealias=dealias)
#xbasis = d3.ComplexFourier(coord['x'], size=Nx, bounds=(0, Lx), dealias=dealias)

#ybasis = d3.ComplexFourier(coord['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

#We now define the fields tha we need to calculate. The velocity field and the orientation field that his a tensor

psi = dist.Field(name='psi', bases=(xbasis, ybasis))  # u=dy(psi), v=-dx(psi)
#Q=dist.TensorField(coords,name='Q',bases=(xbasis,ybasis),components=2)# we comment this it will be solved later
tau_psi = dist.Field(name='tau_psi')

#Rembmer Q is symmetric and traceless we define first as scalar fields
Qxx=dist.Field(name='Qxx',bases=(xbasis,ybasis))

#Qyy=-Qxx
Qxy=dist.ScalarField(name='Qxy',bases=(xbasis,ybasis))

dx = lambda A: d3.Differentiate(A, coord['x'])
dy = lambda A: d3.Differentiate(A, coord['y'])
u = dy(psi)
v = -dx(psi)
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

lambda_n=1#tumbling paramete

#A=lambda*E+Omega
A11=lambda_n*Exx+Omegaxx
A12=lambda_n*Exy+Omegaxy
A21=lambda_n*Exy+Omegayx
A22=lambda_n*Eyy+Omegayy
#B=Q+I/3
B11=Qxx+1/3
B12=Qxy
B21=Qxy
B22=-Qxx+1/3

#C=lambda*E-Omega
C11=lambda_n*Exx-Omegaxx
C12=lambda_n*Exy-Omegaxy
C21=lambda_n*Exy-Omegayx
C22=lambda_n*Eyy-Omegayy
#D=Q:nabla u
D=Qxx*(ux-vy)+Qxy*(uy+vx)
curl_f = dx(dx(Qxy)) - 2*dx(dy(Qxx))  - dy(dy(Qxy))
####################################################################
#We now write explicit expression for S11 and S12
####################################################################

S11=(A11*B11+A12*B21)+(B11*C11+B12*C21)-2*lambda_n*B11*D
S12=A11*B12+A12*B22+(B11*C21+B12*C22)-2*lambda_n*B12*D

# operators
lap = lambda A: dx(dx(A)) + dy(dy(A))

# active force
xi=5.0 #activity parameter
fx = xi * (dx(Qxx) + dy(Qxy))
fy = xi * (dx(Qxy) - dy(Qxx))
problem = d3.IVP([psi,Qxx,Qxy,tau_psi], namespace=locals())
#equation for the components of the nematic tensor field
problem.add_equation("dt(Qxx) = -u*dx(Qxx) - v*dy(Qxx) + S11 + (1/Pe)*lap(Qxx)")
problem.add_equation("dt(Qxy) = -u*dx(Qxy) - v*dy(Qxy) + S12 + (1/Pe)*lap(Qxy)")
#continuity equation for the velocity field
problem.add_equation("lap(lap(psi)) - xi*curl_f +tau_psi= 0")
problem.add_equation("integ(psi) = 0")
#we define the initial conditions for the fields
# Small random perturbation around isotropic state

# --- Initial conditions ---
Qxx['g'] = 1e-3* np.random.randn(Nx, Ny)
Qxy['g'] = 1e-3 * np.random.randn(Nx, Ny)
psi['g'] = 0.0
# --- Solver ---
solver = problem.build_solver(d3.SBDF2)
solver.stop_sim_time = 1.0
timestep = 1e-4

# --- File handler (MUST be before the time loop) ---
snapshots = solver.evaluator.add_file_handler('snapshots', iter=20,max_writes=10000)
snapshots.add_task(Qxx, name='Qxx')
snapshots.add_task(Qxy, name='Qxy')
snapshots.add_task( dy(psi), name='u')
snapshots.add_task(-dx(psi), name='v')
snapshots.add_task(psi,      name='psi')

# --- Time loop ---
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 20 == 0:
        Qxx_max = np.max(np.abs(Qxx['g'].real))
        Qxy_max = np.max(np.abs(Qxy['g'].real))
        psi_max = np.max(np.abs(psi['g'].real))
        print(f"t={solver.sim_time:.5f}  Qxx={Qxx_max:.4e}  Qxy={Qxy_max:.4e}  psi={psi_max:.4e}")
        if np.isnan(Qxx_max) or np.isnan(psi_max):
            print("NaN detected — stopping")
            break
    if solver.iteration % 100 == 0:
            print(f"t = {solver.sim_time:.3f}")
