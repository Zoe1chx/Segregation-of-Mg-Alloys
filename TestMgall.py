import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from pathlib import Path
from scipy import sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd 
import os
directory = 'C:\\Users\\DELL\\Desktop\\script\\datafinal\\Quanzhuding\\test'
if not os.path.exists(directory):
    os.makedirs(directory)

# Output
saveFiguresToFile = True
outputFilename = 'anim'

# Geometry
xmin = 0.
xmax = 0.09
ymin = 0.
ymax = 0.3

# Spatial discretization
dx = 0.0009*100/100
dy = 0.003*100/100

# Initial conditions
T0 = 1000
kp=0.3
# Boundary conditions
# Wall temperature: [W, E, S, N], np.nan = symmetry
Twall = [473, 473, 473, 1100]
Cwall = [0.06*kp, 0.06*kp, 0.06*kp, 0.06]
# Wall x-velocity: [W, E, S, N], np.nan = symmetry
uWall = [0., 0., 0., 0]
# Wall y-velocity: [W, E, S, N], np.nan = symmetry
vWall = [0., 0., 0., 0.]
# Pressure: [W, E, S, N], np.nan = symmetry
pWall = [np.nan, np.nan, np.nan, np.nan]

# Physical constants
g = 9.81
h = 10  #  W/(m²·K)
T_ext = 293 # K

# Material properties
rho = 1740.0
c = 1320.04
k = 85.751215
a = k/(rho*c)
mu = 1.7e-3
nu = mu/rho
beta1 = 2.7e-6
beta2 =0.048
Tref = 300
Tm =590
L = 255000
D = 3e-8
C0=0.06
Cref=0.03
# Model parameters
dTm = 100
Cmush = 0
Tl=641.9+273
Ts=628.443+273
# Temporal discretization
tMax = 1080.
sigma = 0.25
#K0=40e-6
# Solver settings
poissonSolver = 'iterative'
nit = 50 # iterations of pressure poisson equation

# Visualization
dtOut =1# output step length
nOut = int(round(tMax/dtOut))
figureSize = (10., 6.25)
minContour1 = 200.
maxContour1 = 1120.
colormap1 = 'jet'
plotContourLevels1 = np.linspace(minContour1, maxContour1, num=23)
ticks1 = np.linspace(minContour1, maxContour1, num=12)
minContour2 = 0.
maxContour2 = 0.005
colormap2 = 'jet'
plotContourLevels2 = np.linspace(minContour2, maxContour2, num=21)
ticks2 = np.linspace(minContour2, maxContour2, num=11)

# Initial time step calculation
dtVisc = min(dx**2/(2*nu), dy**2/(2*nu))  # Viscous time step
dtThrm = min(dx**2/a, dy**2/a)  # Thermal time step
dt0 = sigma*min(dtVisc, dtThrm)
nsignifdigs = -np.floor(np.log10(abs(dtOut/dt0)))
dt = dtOut/(np.ceil(dtOut/dt0*10**nsignifdigs)/(10**nsignifdigs))

# Mesh generation
nx = int((xmax-xmin)/dx)+1
ny = int((ymax-ymin)/dy)+1
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

# Initial values
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
T = T0*np.ones((ny, nx))
U = np.zeros((ny, nx))
f = np.zeros((ny, nx))
f = f.astype(float)
cMod = np.ones((ny, nx))
B = np.zeros((ny, nx))

CP = np.zeros((ny, nx))
CW = np.zeros((ny, nx))
CE = np.zeros((ny, nx))
CS = np.zeros((ny, nx))
CN = np.zeros((ny, nx))
A = sp.csr_matrix((nx*ny, nx*ny))
C = C0*np.ones((ny, nx))
# Determine the index for x = 0.7

# Assuming x is defined somewhere earlier in your code
x_threshold_index_start = np.where(x > 0.01)[0][0]
x_threshold_index_end = np.where(x < 0.08)[0][-1]  # Corrected to use the last index where x < 0.8

T[:, x_threshold_index_start:x_threshold_index_end + 1] = 1100 # Set temperature to 1100 where x is between 0.1 and 0.8
T[:, :x_threshold_index_start] = 200  # Set temperature to 200 where x < 0.1
T[:, x_threshold_index_end + 1:] = 200  # Set temperature to 200 where x > 0.8

C[:, x_threshold_index_start:x_threshold_index_end + 1] = 0.06  # Set concentration to 0.06 where x is between 0.1 and 0.8
C[:, :x_threshold_index_start] = 0.018  # Set concentration to 0.018 where x < 0.1
C[:, x_threshold_index_end + 1:] = 0.018  # Set concentration to 0.018 where x > 0.8

f[:, x_threshold_index_start:x_threshold_index_end + 1] = 1  # Set f to 1 where x is between 0.1 and 0.8
f[:, :x_threshold_index_start] = 0  # Set f to 0 where x < 0.1
f[:, x_threshold_index_end + 1:] = 0  # Set f to 0 where x > 0.8

# Initialize Concentration C
      # Set concentration to 0.15 * 0.06 for x >
# Find the index where y <= 0.1
# After initializing concentration C for all values
y_threshold_index = np.where(y > 0.025)[0][0]  # Get the first index where y exceeds 0.1
C[:y_threshold_index, :] = kp * 0.06  # Set concentration to kp * 0.06 for y <= 0.1
T[:y_threshold_index, :] = 473 # Set concentration to kp * 0.06 for y <= 0.1
f[:y_threshold_index, :] = 0 
# Functions
K0=1e-4
K=K0*K0/180*(f+1e-3)*(f+1e-3)*(f+1e-3)/(1-f+1e-3)/(1-f+1e-3)
# Initial and final temperatures for the left wall
Twall_initial =1100 # Initial temperature in degrees Celsius
Twall_final = 473    # Final temperature in degrees Celsius
kk=1e-9
def animateContoursAndVelocityVectors():

    fig = plt.figure(figsize=figureSize, dpi=100)

    # plot temperature and pressure
    # Axis
    ax1 = fig.add_subplot(121)
    ax1.set_aspect(1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Temperature and pressure contours')
    # Filled contours for temperature
    ctf1 = ax1.contourf(X, Y, T, plotContourLevels1, extend='both',
                        alpha=1, linestyles=None, cmap=colormap1)
    # Contours for pressure
    ct1 = ax1.contour(X, Y, p, levels=20,
                      colors='black', linewidths=1, linestyles='dotted')
    plt.clabel(ct1, ct1.levels[::2], fmt='%1.1e', fontsize='smaller')
    # Colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True, ticks=ticks1)
    cBar1.set_label('T / °C')

    # Plot liquid fraction and velocity
    ax2 = fig.add_subplot(122)
    ax2.set_aspect(1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Liquid fraction and velocity')
    # Filled contours
    ax2.contourf(X, Y, f, np.linspace(0, 1, num=11), extend='both',
                 alpha=1, linestyles=None, cmap='gray')
    # plot velocity
    m = 1
    qv = ax2.quiver(X,
                    Y,
                    u,
                    v,
                    clim=np.array([min(plotContourLevels2),
                                   max(plotContourLevels2)]),
                    cmap=colormap2)
    # Colorbar
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cBar2 = fig.colorbar(qv, cax=cax2, extendrect=True, ticks=ticks2)
    cBar2.set_label('U / m/s')

    plt.tight_layout()

    plt.show()

    if saveFiguresToFile:
        formattedFilename = '{0}_{1:5.3f}.png'.format(outputFilename, t)
        path = os.path.join(directory, formattedFilename)
        plt.savefig(path)


def solveMomentumEquation(u, v, un, vn, dt, dx, dy, nu, beta1,beta2, T, Tn,Cn, g,C,C0,p,pn):
    Qu = -dt/(dx)*((np.maximum(un[1:-1, 1:-1], 0)*un[1:-1, 1:-1] +
                   np.minimum(un[1:-1, 1:-1], 0)*un[1:-1, 2:]) - (
                   np.maximum(un[1:-1, 1:-1], 0)*un[1:-1, 0:-2] +
                   np.minimum(un[1:-1, 1:-1], 0)*un[1:-1, 1:-1])) + \
        -dt/(dy)*((np.maximum(vn[1:-1, 1:-1], 0)*un[1:-1, 1:-1] +
                  np.minimum(vn[1:-1, 1:-1], 0)*un[2:, 1:-1]) - (
                  np.maximum(vn[1:-1, 1:-1], 0)*un[0:-2, 1:-1] +
                  np.minimum(vn[1:-1, 1:-1], 0)*un[1:-1, 1:-1])) + \
        nu * (
            dt/dx**2*(un[1:-1, 2:]-2*un[1:-1, 1:-1]+un[1:-1, 0:-2]) +
            dt/dy**2*(un[2:, 1:-1]-2*un[1:-1, 1:-1]+un[0:-2, 1:-1]))
    u[1:-1, 1:-1] = un[1:-1, 1:-1] + Qu#-dt*nu/K[1:-1, 1:-1]*un[1:-1, 1:-1]#-Px#-dt*nu/kk*un[1:-1, 1:-1]*f[1:-1, 1:-1]**2
   # -dt*nu/K[1:-1, 1:-1]*un[1:-1, 1:-1]
   

    # Solve v-velocity
    Qv = -dt/(dx)*((np.maximum(un[1:-1, 1:-1], 0)*vn[1:-1, 1:-1] +
                   np.minimum(un[1:-1, 1:-1], 0)*vn[1:-1, 2:]) - (
                   np.maximum(un[1:-1, 1:-1], 0)*vn[1:-1, 0:-2] +
                   np.minimum(un[1:-1, 1:-1], 0)*vn[1:-1, 1:-1])) + \
        -dt/(dy)*((np.maximum(vn[1:-1, 1:-1], 0)*vn[1:-1, 1:-1] +
                  np.minimum(vn[1:-1, 1:-1], 0)*vn[2:, 1:-1]) - (
                  np.maximum(vn[1:-1, 1:-1], 0)*vn[0:-2, 1:-1] +
                  np.minimum(vn[1:-1, 1:-1], 0)*vn[1:-1, 1:-1])) + \
        nu * (
            dt/dx**2*(vn[1:-1, 2:]-2*vn[1:-1, 1:-1]+vn[1:-1, 0:-2]) +
            dt/dy**2*(vn[2:, 1:-1]-2*vn[1:-1, 1:-1]+vn[0:-2, 1:-1]))
    Sv = dt*beta1*g*(T[1:-1, 1:-1]-Tref)+dt*beta2*g*(C[1:-1, 1:-1]-Cref)
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + Qv + Sv#-dt*nu/K[1:-1, 1:-1]*vn[1:-1, 1:-1]#-Py #-dt*nu/kk*vn[1:-1, 1:-1]*f[1:-1, 1:-1]**2#
    return u, v

def calcVelocitySwitchOff(u, v, B, dt, f, Cmush):
    # Velocity switch-off constant
    q = 1e-3
    B[1:-1, 1:-1] = Cmush*(1-f[1:-1, 1:-1])**2/(f[1:-1, 1:-1]**3+q)

    # Velocity switch off
    u[1:-1, 1:-1] = u[1:-1, 1:-1] / (1 + dt * B[1:-1, 1:-1])
    v[1:-1, 1:-1] = v[1:-1, 1:-1] / (1 + dt * B[1:-1, 1:-1])

    # Set velocity to zero where the liquid fraction f is zero (solid phase)
  #  u[f == 0] = 0
    #v[f == 0] = 0

    return u, v, B


def setVelocityBoundaries(u, v, uWall, vWall):
    # West - 
    u[:, 0] = uWall[0]  
    v[:, 0] = vWall[0]  
    #u[:, 0] = u[:, 1]  
    #v[:, 0] = v[:, 1]  
    # East - 
    u[:, -1] = uWall[1]
    v[:, -1] = vWall[1]
   # u[:, -1] = u[:, -2]
   # v[:, -1] = v[:, -2]
    # South -
    u[0, :] = uWall[2]
    v[0, :] = vWall[2]
    #u[0, :] = u[1, :] 
    #v[0, :] = v[1, :]
    # North - 
    u[-1, :] = uWall[3]
    v[-1, :] = vWall[3]
   # u[-1, :] = u[-2, :]
   # v[-1, :] =v[-2, :]
    #u[f==0]=u[f==0]/10
    #v[f==0]=v[f==0]/10
    return u, v


def buildPoissonRightHandSide(b, rho, dt, u, v, dx, dy, beta1,beta2, g, T,C):

    # Right hand side
    b[1:-1, 1:-1] = rho*(
        1/dt*((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) +
              (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))
        - (((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 +
           2*((u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy) *
              (v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) +
           ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2)
        + beta1*g*(T[2:, 1:-1] - T[0:-2, 1:-1])/(2*dy)+ beta2*g*(C[2:, 1:-1] - C[0:-2, 1:-1])/(2*dy))
    #- nu/K[1:-1, 1:-1]*dt*(u[1:-1, 1:-1]+v[1:-1, 1:-1])

    return b


def solvePoissonEquation(p, b, dx, dy, nit, pWall, solver,
                         CP, CW, CE, CS, CN, A, nx, ny):

    if solver == 'direct':

        # Inner nodes
        CP[1:-1, 1:-1] = 2*(1/dx**2+1/dy**2)
        CW[1:-1, 1:-1] = -1/dx**2
        CE[1:-1, 1:-1] = -1/dx**2
        CS[1:-1, 1:-1] = -1/dy**2
        CN[1:-1, 1:-1] = -1/dy**2

        # Boundary conditions
        CP[:, 0] = 1
        CP[:, -1] = 1
        CP[0, :] = 1
        CP[-1, :] = 1
        CW[1:-1, -1] = -1
        CE[1:-1, 0] = -1
        CS[-1, 1:-1] = 0
        CN[0, 1:-1] = -1
        CW[0, -1] = -0.5
        CW[-1, -1] = 0
        CE[0, 0] = -0.5
        CE[-1, 0] = 0
        CS[[-1, -1], [0, -1]] = 0
        CN[[0, 0], [0, -1]] = -0.5

        A = sp.csr_matrix(sp.spdiags((CP.flat[:], CW.flat[:], CE.flat[:],
                                      CS.flat[:], CN.flat[:]),
                                     [0, 1, -1, nx, -(nx)],
                                     ((nx)*(ny)), ((nx)*(ny))).T)

        p.flat[:] = spla.spsolve(A, -b.flat[:], use_umfpack=True)

    elif (solver == 'iterative'):

        for nit in range(50):

            # Reference pressure in upper left corner
            # p[-1, 1] = 0

            p[1:-1, 1:-1] = (dy**2*(p[1:-1, 2:]+p[1:-1, :-2]) +
                             dx**2*(p[2:, 1:-1]+p[:-2, 1:-1]) -
                             b[1:-1, 1:-1]*dx**2*dy**2)/(2*(dx**2+dy**2))

            # Boundary conditions
            # West
            if np.isnan(pWall[0]):
                p[:, 0] = p[:, 1]  # symmetry
            else:
                p[:, 0] = pWall[0]
            # East
            if np.isnan(pWall[1]):
                p[:, -1] = p[:, -2]  # symmetry
            else:
                p[:, -1] = pWall[1]
            # South
            if np.isnan(pWall[2]):
                p[0, :] = p[1, :]  # symmetry
            else:
                p[0, :] = pWall[2]
            # North
            if np.isnan(pWall[3]):
                p[-1, :] = p[-2, :]  # symmetry
            else:
                p[-1, :] = pWall[3]

    return p


def correctPressure(u, v, p, rho, dt, dx, dy, B):

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - 1/rho*dt/(1+dt*B[1:-1, 1:-1]) * \
        (p[1:-1, 2:]-p[1:-1, 0:-2])/(2*dx)
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - 1/rho*dt/(1+dt*B[1:-1, 1:-1]) * \
        (p[2:, 1:-1]-p[0:-2, 1:-1])/(2*dy)

    return u, v


def calcEnthalpyPorosity(T, Tm, dTm, L, c,f):

   # # Calculate liquid phase fraction
   # f = (T-Tm)/dTm+0.5
   # f = np.minimum(f, 1)
   # f = np.maximum(f, 0)

    # Find phase change cells
    pc = np.logical_and(f > 0, f < 1)

    # Set heat capacity modifier
    cMod[pc] = 1+L/(c*dTm)
    cMod[np.logical_not(pc)] = 1
    
    return f, pc, cMod

def calcf(T,Tl,Ts,kp):
   #f = (T-Tm)/dTm+0.5
   #f = np.minimum(f, 1)
   #f = np.maximum(f, 0)

    mask_liquid = T >= Tl
    mask_solid = T <= Ts
    mask_mushy = ~mask_liquid & ~mask_solid  
    # Assign values based on the conditions
    f[mask_liquid] = 1  # Fully liquid
    f[mask_solid] = 0   # Fully solid
    f[mask_mushy] = 1-1/(1-kp)*(T[mask_mushy] - Tl) / (T[mask_mushy] - 648.15-273)  
    #f[C== 0.06 * kp] = 0
    return f

  # f = (T-Tm)/dTm+0.5
   #f = np.minimum(f, 1)
   #f = np.maximum(f, 0)
   


def solveEnergyEquation(T, Tn, u, v, dt, dx, dy, a, Twall, cMod, Tm, f,T_ext,h):
    dfs_dt = ((1-f)-(1-fn))
    S2=L*dfs_dt/c/rho
    # Solve energy equation with modified heat capacity
    T[1:-1, 1:-1] = Tn[1:-1, 1:-1] - 1/cMod[1:-1, 1:-1] * (
        dt/(dx)*(np.maximum(u[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 1:-1]-Tn[1:-1, 0:-2]) +
                 np.minimum(u[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 2:]-Tn[1:-1, 1:-1])) +
        dt/(dy)*(np.maximum(v[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 1:-1]-Tn[0:-2, 1:-1]) +
                 np.minimum(v[1:-1, 1:-1], 0) *
                 (Tn[2:, 1:-1]-Tn[1:-1, 1:-1]))) + \
        a * \
        (dt/dx**2*(Tn[1:-1, 2:]-2*Tn[1:-1, 1:-1]+Tn[1:-1, 0:-2]) +
         dt/dy**2*(Tn[2:, 1:-1]-2*Tn[1:-1, 1:-1]+Tn[0:-2, 1:-1]))+S2[1:-1, 1:-1]
    return T

def applyTemperatureBoundaryConditions(T, Twall, h, T_ext, k, dy, dt, t):
    # 根据时间 t 计算左边界温度
    T[:, 0] =T[:, 1] + h * (T_ext - T[:, -2]) * dt / (k * dy)
    #T[:, 0] = Twall[0]  # temperature, at xmin 
    # 对流换热边界条件，右边界
    if not np.isnan(Twall[1]):
        T[:, -1] = T[:, -2] + h * (T_ext - T[:, -2]) * dt / (k * dy)   
    # 对流换热边界条件，底边界
    if not np.isnan(Twall[2]):
        T[0, :] = T[1, :] + h* (T_ext - T[1, :]) * dt / (k * dy)    
    # 对流换热边界条件，顶边界
    T[-1, :] = T[-2, :] #+ h* (T_ext - T[1, :]) * dt / (k * dy)
    #Twall[3]
    #721.7653 - 2.861 * t + 0.0183 * t**2 - 3.89e-5 * t**3
   # T[50,50]=721.7653 - 2.861 * t + 0.0183 * t**2 - 3.89e-5 * t**3
    return T

def applyConcentrationBoundaryConditions(C, T, f, Cwall, dx, dy, dt,L,rho):
   C[:, 0] = C[:, 1] #左边界
   C[:, -1] =C[:, -2] #右边界           
   C[0, :] = C[1, :] # 下边界   
   C[-1, :]=C[-2, :] # 上边界
   #C[:, 0] = 0.06*kp#左边界
   #C[:, -1] =0.06*kp#右边界           
   #C[0, :] = 0.06*kp## 下边界   
   #C[-1, :]=0.06 # 上边界        
   return C
      
def solveConcentrationEquation(C, Cn, u, v, dt, dx, dy, D,f,fn,kp):
    
   # dfs_dt =Cn[1:-1, 1:-1]*((1-fn[1:-1, 2:])+(1-fn[1:-1, 0:-2])+(1-fn[2:, 1:-1])+(1-fn[0:-2, 1:-1])-4*(1-fn[1:-1, 1:-1]))
    dfs_dt = (C[1:-1, 1:-1]*(1-f[1:-1, 1:-1])-C[1:-1, 1:-1]*(1-fn[1:-1, 1:-1])) #Cs*dfs
    S = (1 - kp) * dfs_dt #(1-kp)*Cs*dfs
    C[1:-1, 1:-1] = Cn[1:-1, 1:-1] - (
        dt/(dx)*(np.maximum(u[1:-1, 1:-1], 0) *
                 (Cn[1:-1, 1:-1]-Cn[1:-1, 0:-2]) +
                 np.minimum(u[1:-1, 1:-1], 0) *
                 (Cn[1:-1, 2:]-Cn[1:-1, 1:-1])) +
        dt/(dy)*(np.maximum(v[1:-1, 1:-1], 0) *
                 (Cn[1:-1, 1:-1]-Cn[0:-2, 1:-1]) +
                 np.minimum(v[1:-1, 1:-1], 0) *
                 (Cn[2:, 1:-1]-Cn[1:-1, 1:-1]))) + \
        D*f[1:-1, 1:-1]* \
        (dt/dx**2*(Cn[1:-1, 2:]-2*Cn[1:-1, 1:-1]+Cn[1:-1, 0:-2]) +
         dt/dy**2*(Cn[2:, 1:-1]-2*Cn[1:-1, 1:-1]+Cn[0:-2, 1:-1]))+S/(f[1:-1, 1:-1]+1e-2)/kp*dt
    #扩散系数加权
    return C


def visualizeConcentration(X, Y, C):
    fig, ax = plt.subplots()
    cont = ax.contourf(X, Y, C, 10, cmap='jet')  # Plotting the concentration field
    fig.colorbar(cont, ax=ax, label='Concentration')  # Adding a colorbar for reference
    ax.set_xlabel('x (m)')  # Label for the x-axis
    ax.set_ylabel('y (m)')  # Label for the y-axis
    ax.set_title('Concentration distribution')  # Title for the plot
    
    # Setting the axes' limits based on the ranges of X and Y
    ax.set_xlim(X.min(), X.max())  # X-axis limits to match the grid
    ax.set_ylim(Y.min(), Y.max())  # Y-axis limits to match the grid
    
    # Setting the aspect ratio to be equal to ensure that one unit in x is equal to one unit in y
    ax.set_aspect('equal')
    
    plt.show()


# Create figure
plt.close('all')

# Time stepping
twall0 = time.time()
tOut = dtOut
t = 0
n = 0
uMax = 0
vMax = 0
def save_to_excel_grid(t, T, p, u, v, f, C, base_directory):
    # 时间点格式化，将用于文件命名
    time_stamp = f"{int(t):04d}"
    
    # 数据字典，键为变量名，值为对应的数据
    data_dict = {
        'Temperature': T,
        'Pressure': p,
        'U_velocity': u,
        'V_velocity': v,
        'Liquid_fraction': f,
        'Concentration': C
    }
    
    # 为每个变量创建一个单独的Excel文件
    for variable_name, data in data_dict.items():
        # 文件名包含变量名和时间戳
        filename = os.path.join(base_directory, f"{variable_name}_{time_stamp}.xlsx")
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet(variable_name)  # 创建工作表
        
        # 写入数据
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                worksheet.write(row, col, data[row, col])
        
        # 关闭工作簿
        workbook.close()

# 示例调用（假设在循环中）
directory = 'C:\\Users\\DELL\\Desktop\\script\\datafinal\\Quanzhuding\\test'
base_directory= 'C:\\Users\\DELL\\Desktop\\script\\datafinal\\Quanzhuding\\test'
if not os.path.exists(directory):
    os.makedirs(directory)

save_to_excel_grid(t, T, p, u, v, f, C, directory)
while t < tMax:

    n += 1
    
    # Automatic time stepping
    #if np.logical_and(uMax > 0, vMax > 0):
      # dtConv = min(dx/uMax, dy/vMax)
      # dtVisCon = 1/(1/dtVisc+1/dtConv)
      # dt0 = sigma*min(dtThrm, dtVisCon)
      # nsigdigs = -np.floor(np.log10(abs(dtOut/dt0)))
     #  dt = dtOut/(np.ceil(dtOut/dt0*10**nsigdigs)/(10**nsigdigs))
     #  dt = min(dt, tOut-t)
     #渗透 #没考虑
    dt=5e-3
    t += dt

    # Update variables
    pn = p.copy()
    un = u.copy()
    vn = v.copy()
    fn = f.copy()
    Tn = T.copy()
    Cn = C.copy()
    f=calcf(T,Tl,Ts,kp)
    
    C = solveConcentrationEquation(C, Cn, u, v, dt, dx, dy, D,f,fn,kp)
    C= applyConcentrationBoundaryConditions(C, T, f, Cwall, dx, dy, dt,L,rho)
    [f, pc, cMod] = calcEnthalpyPorosity(T, Tm, dTm, L, c,f)
    # Momentum equation with projection method
    # Intermediate velocity field u*
    [u, v] = solveMomentumEquation(u, v, un, vn, dt, dx, dy,
                                   nu, beta1,beta2 ,T, Tref,Cref, g,C,C0,p,pn)
    # Switch off velocities at solid phase
    [u, v, B] = calcVelocitySwitchOff(u, v, B, dt, f, Cmush)
    # Set velocity boundaries
    [u, v] = setVelocityBoundaries(u, v, uWall, vWall)
    # Pressure correction
    b = buildPoissonRightHandSide(b, rho, dt, u, v, dx, dy, beta1,beta2, g, T,C)
    p = solvePoissonEquation(p, b, dx, dy, nit, pWall, poissonSolver,
                             CP, CW, CE, CS, CN, A, nx, ny)
    [u, v] = correctPressure(u, v, p, rho, dt, dx, dy, B)
    [u, v] = setVelocityBoundaries(u, v, uWall, vWall)
    # Energy equation
    # In the main simulation loop
    
    T = solveEnergyEquation(T, Tn, u, v, dt, dx, dy, a, Twall, cMod, Tm, f,T_ext,h)
    T = applyTemperatureBoundaryConditions(T, Twall, h, T_ext, k, dy, dt, t)
    # Enthalpy porosity method
    
    
    # Output
    if (t-tOut) > -1e-6:

        t = tOut
        

        # Calculate derived quantities
        # Velocities
        U = (u**2+v**2)**0.5
        uMax = np.max(np.abs(u))
        vMax = np.max(np.abs(v))
        # Peclet numbers
        Pe_u = uMax*dx/nu
        Pe_v = vMax*dy/nu
        # Courant Friedrichs Levy numbers
        CFL_u = uMax*dt/dx
        CFL_v = vMax*dt/dy
        # Rayleigh numbers
        if np.any(f > 0):
            dTmaxLiq = np.max(T[f > 0])-np.min(T[f > 0])
        else:
            dTmaxLiq = 0
        RaW = g*beta1*dTmaxLiq*(xmax-xmin)**3/(nu*a)
        RaH = g*beta1*dTmaxLiq*(ymax-ymin)**3/(nu*a)

        print("==============================================================")
        print(" Time step n = %d, t = %8.3f, dt = %4.1e, t_wall = %4.1f" %
              (n, t, dt, time.time()-twall0))
        print(" max|u| = %3.1e, CFL(u) = %3.1f, Pe(u) = %4.1f, RaW = %3.1e" %
              (uMax, CFL_u, Pe_u, RaW))
        print(" max|v| = %3.1e, CFL(v) = %3.1f, Pe(v) = %4.1f, RaH = %3.1e" %
              (vMax, CFL_v, Pe_v, RaH))

        animateContoursAndVelocityVectors()
    
        save_to_excel_grid(t, T, p, u, v, f, C, base_directory)
        visualizeConcentration(X, Y, C)
        tOut += dtOut