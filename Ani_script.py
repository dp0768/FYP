import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from scipy.constants import e, m_n, m_p, pi, epsilon_0, hbar, m_u
import scipy.interpolate as interpolate
import os
import warnings
import pandas as pd
import numba
from numba import jit
import time

# I apologise to whomever comes across this disgustingly laid out script.
# Doing my BSc final year project meant the script kept changing a lot as I developed it eventually reaching this abomination.

os.chdir(os.getcwd()+"\FYP\CODE")

warnings.filterwarnings("ignore")  # suppress numpy ComplexWarning

#Show graphs
FILM_1 = True
FILM_2 = False

#Plotting colors
# Experimental - circles, Black, no  connecting line
# Analytical - Blue, solid Line
# Numerical - Red, Dashes


#Redefine Planck constant to MeV 10^-22 seconds (0.1*zs)
hbar = (hbar/(e*10**6))/10**-22

#Redefine Permitivity to F fm^-1 
epsilon_0 = epsilon_0 / 10**15
#HMC MeV^-1 fm^-2
HMC = 0.04818696

# Radial grid setup (fm)
r_min = 0.1
r_max = 120
dr = 0.1
nr = int((r_max-r_min)/dr)
r = np.linspace(r_min,r_max,nr)

#Time Grid setup 10^-22 seconds
#Actual time steps used in simulations (1e-7 seems optimal)
dt = 1e-7
# number of time steps between actually saving data (always 10^3 less than nt if dt used)
nt_skip = round((dt)**-1 / 1000)
t_max = 60
t_min = 0 
nt = int((t_max-t_min)/dt)
nt_save = int(nt/nt_skip)
t_save = np.linspace(t_min,t_max,nt_save)

Psi_save = np.zeros(shape=[nr])
Psi_new = np.zeros(shape=[nr])

#Energy Array (MeV)
E_min = 0
E_max = 100
nE = 10**6
dE = (E_max-E_min)/nE
E = np.linspace(E_min,E_max,nE)

# Particle 1 - Oxygen
p_num_P = 8
n_num_P = 8

m_P = (p_num_P * m_p) + (n_num_P * m_n)
q_P = 8 * e
A_P = 16

# Particle 2 - Samarium
p_num_T = 62
n_num_T = 144-62

m_T = (p_num_T * m_p) + (n_num_T * m_n)

q_T = 62 * e
A_T = 144

#Reduced Mass (dimensionless)
mu_macron = (A_P * A_T) / (A_T + A_P)
mu = mu_macron * m_u

#Wood-Saxon constants (it is just easier to have them here rather than inside function :) )
# Max Potential (MeV)
V_0 = 105.1
# Length (fm) 
a_0 = 0.75
#(fm)
r_0 = 1.1 
# Nuclear Radius (fm) 
R = r_0 * (A_P**(1/3)+A_T**(1/3))

def Finite_Diff (f_x,dx,nx,step=False,x=0):
    Second_Derivative = np.zeros(nx)
    #Finite difference for function with equal dx
    if step == False:
        for i in range (1,nx-1):
                Second_Derivative[i] = (f_x[i+1]-2*f_x[i]+f_x[i-1]) / (dx**2)
    
    #finite difference for function with unequal in        
    if step == True:
        for i in range (1,nx-1):
            Second_Derivative[i] =  2 * ( ((f_x[i+1]-f_x[i]) / (x[i+1]-x[i])) - ((f_x[i]-f_x[i-1]) / (x[i]-x[i-1])) ) * (1/(x[i+1]-x[i-1]))
        
    return Second_Derivative[1:-1]

def Trapezoid_integration(f_x,dx,nx):
    Integral = 0
    for i in range(1,nx):
        a = (f_x[i-1]+f_x[i])*(dx/2)
        Integral = Integral + a
    return Integral

def V_calc(Coulomb=False, Analysis=False):
    #global txt
    #Wood-Saxon constants
    # Max Potential (MeV)
    V_0 = 105.1
    # Length (fm) 
    a_0 = 0.75
    #(fm)
    r_0 = 1.1 
    # Nuclear Radius (fm) 
    R = r_0 * (A_P**(1/3)+A_T**(1/3))

    #define potential arrays
    V_C = np.zeros(nr)
    V_N = np.zeros(nr)
    
    #determine potential values at r#
    for i in range (nr):
        V_C[i] = (1/(4*pi*epsilon_0)) * q_P * q_T / r[i] * (1/e) #ev       
        V_N[i] = - (V_0*10**6) / (1 + np.exp((r[i] - R)/a_0)) #eV
        
    V_total = (V_C + V_N) / 10**6 #MeV
    
    if Coulomb == True:
        return V_C / 10**6
    elif Analysis == True:
        return V_total, V_C/10**6 , V_N/10**6
    else:
        return V_total

def V_analysis(SIM=False, i_V=False):
    #global txt
    V_total, V_C, V_N = V_calc(Analysis=True)
    
    #Determine First Turning point (Minima)
    for i in range(len(V_total)):
        if V_total[i] < V_total[i+1]:
            V_total_min = i
            break
        
    if i_V == True:
        return V_total_min

    #Determine Second Turning point (Maxima)
    for i in range(len(V_total)):
        j = i + V_total_min
        if V_total[j] > V_total[j+1]:
            V_total_max = j
            break
    
    if SIM == True:
        return V_total_max 
    
        
    ###Determine the Curvature of the Fusion Barrier###
    V_der = abs((V_total[V_total_max+1]-2*V_total[V_total_max]+V_total[V_total_max-1]) / (dr**2)) #Central Finite diff
 
    #Curvature of the barrier
    Curv = np.sqrt(2 * V_der/(mu_macron * HMC)) #MeV
    
    
    ###Transmission Probability - Hill wheeler formula###
    T0_E = np.zeros(nE)
    Flag = False
    for i in range(nE):
        T0_E[i] = 1 / (1+np.exp(-(2*np.pi)*(E[i]-V_total[V_total_max])/Curv))
        if T0_E[i] >= 0.5 and Flag == False:
            V_50 = E[i]
            Flag = True
   
   #Wong Analytical formula for Fusion Cross section
    Sigma0_Wong = np.zeros(nE)
    for i in range(nE):
        Sigma0_Wong[i] = ( (Curv * ((r[V_total_max])**2) / (2*E[i]) ) * (np.log(1+np.exp((2*np.pi)*(E[i]-V_total[V_total_max])/Curv))) )*10
   
            
    ###Experimental Nuclear Cross Section data (MeV,mb,mb) (physics Review)
    Sigma0_exp_energies = np.array([62.88,63.38,63.88,64.38,64.88,65.38,65.88,66.38,66.88,67.38,67.88,68.38,68.88,69.38,69.88,70.38,70.88,71.38,71.88,72.88,73.88,74.88,75.88,77.88,79.88,84.88,89.99,99.88]) * ((A_T) / (A_T+A_P))
    Sigma0_exp = np.array([0.15,0.33,0.45,1.5,2.7,5.5,10.2,17.5,28.6,41.3,55.5,71.2,90.6,108,131,150,169,184,208,253,295,348,383,469,552,700,876,1076])
    Sigma0_exp_error = np.array([0.08,0.08,0.08,0.2,0.3,0.3,0.4,0.3,0.3,0.4,0.4,0.5,0.6,1,1,1,1,1,1,2,2,2,2,3,3,5,4,15])        


def V_imag(V_Re):
    #Constants for Imaginary potential trapping wavepacket within nucleus
    a_0 = 0.2 #fm
    W_0 = 50 #MeV
    r_minima = r[V_analysis(i_V=True)] #fm
    
    V_Imag = -1j * (W_0) / (1 + np.exp((r-r_minima) / (a_0)))
    V_SIM = np.array(object=(V_Re + V_Imag), dtype=complex)
    
    return V_SIM

@numba.jit("c16[:,:](c16[:,:], c16[:], c16[:], c16[:])", nopython=True, nogil=True)
def FD_TDSE(Psi, Psi_save, Psi_new, V_SIM):
    Psi_save = Psi[0,:]
    
    for j in range(0,nt_save-1):
        for k in range(0,nt_skip):
            if k == (nt_skip-1):
                for i in range(1,nr-1):
                    Psi[j+1,i] = Psi_new[i] = (Psi_save[i] - (1/(1j*mu_macron*HMC*hbar))*(dt/(dr**2) )*(Psi_save[i+1]- 2*Psi_save[i] + Psi_save[i-1]) + (dt/(hbar*1j))*V_SIM[i]*Psi_save[i])
                    Psi_save[i] = Psi_new[i]
                
                #norm = np.sum(np.absolute(Psi_save)**2*dr)
                #for i in range(1,nr-1):
                #    Psi_save[i] = Psi_save[i]/norm
                #    Psi[j+1,i] = Psi[j+1,i]/norm
                
            else:
                for i in range(1,nr-1):
                    Psi_new[i] = (Psi_save[i] - (1/(1j*mu_macron*HMC*hbar))*(dt/(dr**2) )*(Psi_save[i+1]- 2*Psi_save[i] + Psi_save[i-1]) + (dt/(hbar*1j))*V_SIM[i]*Psi_save[i])
                    Psi_save[i] = Psi_new[i]

                #norm = np.sum(np.absolute(Psi_save)**2*dr)
                #for i in range(1,nr-1):
                #    Psi_save[i] = Psi_save[i]/norm
    
        print(np.round(100*(j/(nt_save-1)),0))

    print("Sim complete")
    return Psi

def Main():
    #global txt
    
    Psi_init = np.zeros(nr)
    Sigma0 = 20 #(fm)
    r0 = 70  #(fm)
    
    #Retrieve Coulomb Potential
    V_C = V_calc(Coulomb=True) #MeV
    #Determine Total Real Potential of system
    V_total = V_calc() #MeV
    #determine and modify potential to include imaginary Wood-Saxon like potential
    V_SIM = V_imag(V_Re=V_total) #MeV
    
    
    #Initialize Psi_nought + set the first and final values to ZERO
    Psi_init = np.exp((-0.5) * ((r-r0)**2) / (Sigma0**2) )
    Psi_init[0] = Psi_init[-1] = 0
    
    #Determine and Normalise Wavefunction 
    norm = 1 / np.sqrt(np.sum(np.absolute(Psi_init)**2 * dr))
    Psi_init = norm * Psi_init

    #Array of Energies to simulate over
    #Emax = 70
    #Emin = 55
    #n = int(Emax-Emin) + 1    
    #E_SIM = np.linspace(Emin,Emax,n,endpoint=True,dtype=int)
    
    n = 1
    E_SIM = np.array([62])

    #Loop to simulate system over range at energies
    for k in range(n):
        start_time = time.time()
        #Define Wavefunction arrays
        Psi_rt = np.zeros(shape=[nt_save,nr],dtype=complex)
        Psi_save = np.zeros(shape=[nr])
        Psi_new = np.zeros(shape=[nr])
        
        #Input Constants/Data
        E0 = E_SIM[k] #(Mev)
        
        #Determine the Wavepackets initial Momentum (K_nought)
        V_C_macron = np.sum((np.absolute(Psi_init)**2) * V_C * dr)
        k0 = np.sqrt( (mu_macron*HMC) * (E0 - (1 / (HMC*mu_macron*(Sigma0**2))) - V_C_macron) )
        
        #Re-initialize Wavefunction with imaginary part (exponential containing momentum)
        for i in range(1,nr-1):
            Psi_rt[0,i] = (np.exp(-(1j*(k0)*r[i]))) * (np.exp( (-0.5) * ((r[i]-r0)**2) / (Sigma0**2)))

        #Determine and Normalise Wavefunction 
        norm = 1/ np.sqrt(np.sum(np.absolute(Psi_rt)**2*dr))
        Psi_rt = norm * Psi_rt
            
        #NUMBA Solver solving TDSE using finite difference method
        Psi_FD_Sol = FD_TDSE(Psi_rt.astype(complex), Psi_save.astype(complex), Psi_new.astype(complex), V_SIM.astype(complex))
        Psi_FD_Sol[0,:] = Psi_rt[0,:]
        
        #Determine Simulation Run time
        sim_time = time.time()-start_time
        
        print(np.imag(V_SIM))

        if FILM_1 == True:
            def init():  
                line.set_data([], [])
                line2.set_data([], [])
                return (line,line2) 
    
            def animate(i):
                y1 = np.real(Psi_FD_Sol[i,:])
                line.set_data(r, y1)
                
                y2 = np.imag(Psi_FD_Sol[i,:])
                line2.set_data(r,y2)
                
                time = np.round(t_save[i], 1)
                tx.set_text("$^{144}Sm + ^{16}O$ Single Barrier Model Gaussian WavePacket Animation \n $\ell=0$, $E_{0}=62$ MeV, $\sigma_{0}=10$ fm \n$t = $"+"{:.1f}".format(time)+' $10^{-22}$ s')
                print("Psi Animation: "+str(np.round(100*(i/(len(t_save))),0))+" %")
                return (line, line2)
    
            ###Animation###
            #plt.rcParams['animation.ffmpeg_path'] = (r"c:\ffmpeg")
            plt.rcParams.update({'font.size': 36})
            plt.rcParams.update({'lines.linewidth' : 4})
            #plt.rcParams.update({'figure.dpi' : 400})
            fig, ax = plt.subplots(1,1,figsize=(30,20))
            ax.set_ylim(bottom=-0.4,top=0.4)
            ax.set_xlim(left=1,right=80)
            tx = ax.set_title("Gaussian Wavepacket Animation \n$t = 0 $10^{-22}$")
            line, = ax.plot([], [], lw = 1.5, label='$Re[\Psi(r)]$',color='blue')
            line2, = ax.plot([], [], lw = 1.5, label ='$Im[\Psi(r)]$',color='red')
            ax.set_ylabel('$\Psi(r)$')
            ax.set_xlabel('Radius (fm)')
            
            ax2 = ax.twinx()
            V_real = ax2.plot(r,V_total,color='green',label='Real Potential')
            V_abs = ax2.plot(r,np.imag(V_SIM),color='green',ls='--',label='Absorption Potential')
            ax2.set_ylim(bottom=-70,top=70)
            ax2.set_ylabel('Energy (MeV)')
            
            # added these three lines
            Lines, Labels = ax.get_legend_handles_labels()
            Lines2, Labels2 = ax2.get_legend_handles_labels()
            ax2.legend(Lines + Lines2, Labels + Labels2, loc=0)
            
            #30 frames per second
            fps = 120
            # number of frames to be used
            frame_num =  int(len(t_save)/(t_max*fps) *1/0.5)
            #Total number of frames, 1 second contains 30 frames and reflects 1x10^-22 seconds in sim
            Frames = np.arange(0, len(t_save), frame_num, dtype=int)
            anim = animation.FuncAnimation(fig, animate, init_func = init, frames = Frames, interval = 20, blit = True) 
            FFwriter  = animation.FFMpegWriter(fps=fps,bitrate=-1)
            anim.save(f'E{E0}_Sigma{Sigma0}.mp4',  writer = FFwriter, dpi=200)
            
        if FILM_2 == True:
            def init():  
                line.set_data([], []) 
                return line, 
    
            def animate(i):
                y1 = np.abs(Psi_FD_Sol[i,:])**2
                line.set_data(r, y1)
                time = np.round(t_save[i], 1)
                tx.set_text("Gaussian Wavepacket Animation PDF \n$t = $"+"{:.1f}".format(time)+' $10^{-22}$')
                print("Psi PDF Animation: "+str(np.round(100*(i/(len(t_save))),0))+" %")
                return line, 
    
            ###Animation###
            #plt.rcParams['animation.ffmpeg_path'] = (r"c:\ffmpeg")
            plt.rcParams.update({'font.size': 36})
            plt.rcParams.update({'lines.linewidth' : 4})
            fig, ax = plt.subplots(1,1,figsize=(30,20))
            ax.set_ylim(bottom=0,top=0.4)
            ax.set_xlim(left=1,right=r_max)
            tx = ax.set_title("Gaussian Wavepacket Animation \n$t = 0 $10^{-22}$")
            line, = ax.plot([], [], lw = 1.5)
            ax.set_ylabel('$|\Psi(r)|^{2}$')
            ax.set_xlabel('Radius (fm)')
            ax2 = ax.twinx()
            ax2.plot(r,V_total,color='green',label='Nuclear Potential')
            ax2.set_ylim(bottom=0,top=150)
            ax2.set_ylabel('Energy (MeV)')
        
            #30 frames per second
            fps = 120
            # number of frames to be used
            frame_num =  int(len(t_save)/(t_max*fps) *1/0.5)   
            #Total number of frames, 1 second contains 30 frames and reflects 1x10^-22 seconds in sim
            Frames = np.arange(0, len(t_save), frame_num, dtype=int)
            anim = animation.FuncAnimation(fig, animate, init_func = init, frames = Frames, interval = 20, blit = True) 
            FFwriter  = animation.FFMpegWriter(fps=fps)
            anim.save(f'E{E0}_Sigma{Sigma0}_PDF.mp4',  writer = FFwriter, dpi=200) 


#V_analysis()
#Main()
