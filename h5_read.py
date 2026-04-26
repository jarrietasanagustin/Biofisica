#Program to read the h5 file and generate the plots of differet 
#data of the active nematics simulations
#Jorge Arrieta April 2026

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os   

Lx=1.0
Ly=1.0
Nx=256
Ny=256
x = np.linspace(0, Lx, Nx, endpoint=False)#this is the x grid, we use endpoint false to avoid the last point since it is periodic  
y = np.linspace(0, Ly, Ny, endpoint=False)#as a for x
[X,Y]=np.meshgrid(x,y,indexing='ij')
# Open the file
with h5py.File('snapshots/snapshots_s1.h5', 'r') as f:
    print("Tasks available:", list(f['tasks'].keys()))
    print("Scales available:", list(f['scales'].keys()))
    
    # Read the data
    Qxx = f['tasks/Qxx'][:]   # shape: (n_writes, Nx, Ny)
    Qxy = f['tasks/Qxy'][:]
    u   = f['tasks/u'][:]
    v   = f['tasks/v'][:]
    t   = f['scales/sim_time'][:]

    print("Data shapes:")
    print("Qxx shape:", np.shape(Qxx))
    print("Qxy shape:", np.shape(Qxy))
    print("u shape:", np.shape(u))
    print("v shape:", np.shape(v))
    print("t shape:", np.shape(t))
    S=np.sqrt(np.real(Qxx)**2 + np.real(Qxy)**2)  # Scalar order parameter
    # Plot the scalar order parameter S at the last time step
    # --- Director field ---
    theta = 0.5 * np.arctan2(Qxy[-1].real, Qxx[-1].real)
    nx = np.cos(theta)
    ny = np.sin(theta)
    # Unwrap to remove branch cut jumps
    theta_unwrap = np.unwrap(np.unwrap(theta, axis=0), axis=1)

    dtheta_dx = np.gradient(theta_unwrap, x, axis=0)
    dtheta_dy = np.gradient(theta_unwrap, y, axis=1)
    grad_theta = np.sqrt(dtheta_dx**2 + dtheta_dy**2)
# --- Subsample for quiver (every n points) ---
    n = 8   # plot every 8 grid points
    xs = X[::n, ::n]
    ys = Y[::n, ::n]
    nxs = nx[::n, ::n]
    nys = ny[::n, ::n]
    #plot.figure(figsize=(6,5))
    #plt.contourf(X, Y, S[-1,:,:],levels=100, cmap='jet')
    #plt.colorbar(label='Scalar Order Parameter S')
 #   plt.title('Scalar Order Parameter S at t={:.2f}'.format(t[-1]))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.contourf(X, Y, S[-1,:,:] , cmap='jet', levels=100)
    plt.colorbar(im, ax=ax, label='Gradient of Director Field')
    # Overlay: director field
# headwidth=0 and headlength=0 makes symmetric arrows (no arrowhead)
# since director has no preferred direction (n = -n)
    ax.quiver(xs, ys,  nxs,  nys, 
          color='white', 
          scale=25, 
          headwidth=0, 
          headlength=0,
          headaxislength=0,
          pivot='middle')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('defects.png', dpi=300)
    plt.show()  