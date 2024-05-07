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
# Note: Code does not iterate over an array of Sigma_nought values, must be manually changed after each set of simulations.


os.chdir(os.getcwd()+"\FYP\CODE")

warnings.filterwarnings("ignore")  # suppress numpy ComplexWarning

#Show graphs
Show_potential = False
Show_transmission = False
Show_cross_section = False
Show_Carrier_dist = False
Sim_Analysis = False
Sim_Analysis_2 = False
FILM_Psi_real= False
FILM_Psi_PDF= False
Show_init_psi = False
Norm_analysis = False 

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

            
    ### Plotting ###
    if Show_potential == True:
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 36})
        plt.rcParams.update({'lines.linewidth' : 4})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        #ax.title.set_text("Total Nuclear Potential")
        #ax.plot(r, V_C, ls = '--', label='Coulomb')
        #ax.plot(r, V_N, ls = '--', label='Nuclear')
        ax.plot(r, V_total, label='Total',color='blue')
        ax.axvline(R, ls='--',label='Nuclear Radius')
        ax.axvline(r[V_total_min], ls='--',color='red', label='Potential trough')
        ax.axvline(r[V_total_max], ls='--',color='Green',label='Barrier peak')
        ax.set_ylabel('Energy (MeV)')
        ax.set_xlabel('Radial separation (fm)')
        #ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylim(bottom=0,top=100)
        ax.set_xlim(left=3,right=20)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.grid()
        plt.savefig('Potentials_HIGH.png',dpi=200,bbox_inches='tight',)
        #plt.show()

    if Show_transmission == True: 
        #First access experimental data
        E_num, T0_num = np.loadtxt('FusProb.dat',unpack=True)
        
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 12})
        font = 12
        FONT = 18
        plt.rcParams.update({'lines.linewidth' : 1.5})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        ax.title.set_text("Transmission probability")
        ax.semilogy(E, T0_E, label='Analytical  - Hill Wheeler',color='blue',ls='-')
        ax.semilogy(E_num, T0_num, ls='--',color='red', label='Numerical - TISE')
        ax.hlines(y=0.5,xmin=0,xmax=V_50, ls='-',label='T_0(E) = 0.5',color='blue')
        ax.vlines(x=V_50,ymin=0,ymax=0.5, ls='-',color='blue')
        ax.set_ylabel('Transmission Probability, $T_{0}(E)$', fontsize=font)
        ax.set_xlabel('Energy (MeV)', fontsize=font)
        ax.set_ylim(bottom=0.001,top=1)
        ax.set_xlim(left=55,right=70)
        plt.legend()
        plt.grid()
        plt.show()
  
    if Show_cross_section == True:
        ###Fusion Cross Section###
        #1fm^2 = 10 mb
    
        #Theory Numerical
        Sigma0_l_sum = np.zeros(nE)
        j = 0
        for i in range(nE):
            Sigma0_l_sum[i] = ( (np.pi * (2*j + 1) * T0_E[i]) / (HMC * mu_macron * E[i]) ) * (E[i] - (j*(j+1)) / (HMC * mu_macron * (r[V_total_max]**2)) ) * 10 #mb
        
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 36})
        plt.rcParams.update({'lines.linewidth' : 8})
        plt.rcParams.update({'figure.dpi' : 100})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        #ax.title.set_text("Nuclear Cross Section - $l=0$ ")
        ax.semilogy(E, Sigma0_Wong, color='blue',ls='-', label='Analytical - Wong')
        ax.errorbar(Sigma0_exp_energies, Sigma0_exp, yerr=Sigma0_exp_error,elinewidth=4,fmt='.k',capsize=10,label='Experimental',markersize=20)
        ax.set_ylabel('Cross Section (mb)')
        ax.set_xlabel('Energy (MeV)')
        ax.set_xlim(left=50,right=80)
        ax.set_ylim(top=10**3,bottom=10**-2)
        plt.legend(loc='lower right')
        plt.grid()
        #plt.subplots_adjust(bottom=0.12,right=0.983,top=0.964,left=0.098)
        plt.savefig('Cross_Section_High.png',dpi=200,bbox_inches='tight',)
        #plt.show()   
        
    if Show_Carrier_dist == True:
        ###Carrier Distributions###
        CS = interpolate.CubicSpline(Sigma0_exp_energies,(Sigma0_exp*Sigma0_exp_energies))
        CS_err = interpolate.CubicSpline(Sigma0_exp_energies,Sigma0_exp_error)
        nBD = len(Sigma0_exp_energies)
        dE_BD = 2*0.9
        Barrier_dist_exp = np.zeros(nBD)
        Barrier_dist_exp_Err = np.zeros(nBD)
        for i in range(0,nBD):
            E_eval = Sigma0_exp_energies[i]
            Barrier_dist_exp[i] = ( ( CS(E_eval+dE_BD) - 2*CS(E_eval) + CS(E_eval-dE_BD) ) / (dE_BD**2) ) 
            Barrier_dist_exp_Err[i] = (E_eval/(dE_BD**2))*np.sqrt( (CS_err(E_eval+dE_BD))**2 + (4*CS_err(E_eval)**2) + (CS_err(E_eval-dE_BD))**2 )

        
        #Theoretical Determination of the Carrier Distributions
        Barrier_dist=np.zeros(nE)
        for i in range(0,nE-1):
            Barrier_dist[i] = ((T0_E[i+1]-T0_E[i] )/dE) * np.pi * r[V_total_max]**2 * 10
    
        #Barrier_dist_wong = np.zeros(nE)
        #Barrier_dist_wong2 = np.zeros(nE)
        #Sigma0_Wong = E*Sigma0_Wong
        #for i in range(0,nE-1):
        #    Barrier_dist_wong[i] = 10 * (2 * np.pi **2 * r[V_total_max]**2 * np.exp((2*np.pi/Curv) * (E[i]-V_total[V_total_max]))) / (Curv * (1 + np.exp((2*np.pi/Curv) * (E[i]-V_total[V_total_max])))**2)
        #    Barrier_dist_wong2[i] = (Sigma0_Wong[i+1] - 2*Sigma0_Wong[i] + Sigma0_Wong[i-1])/(dE**2)       
            
        
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 36})
        plt.rcParams.update({'lines.linewidth' : 8})
        #plt.rcParams.update({'figure.dpi' : 200})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        #ax.title.set_text("Barrier Distribution")
        ax.errorbar(Sigma0_exp_energies,Barrier_dist_exp, yerr=Barrier_dist_exp_Err, elinewidth=4,fmt='.k',capsize=10,label='Experimental',markersize=20)
        ax.plot(E,Barrier_dist,label='Analytical', color='blue',ls='-')
        #ax.plot(E,Barrier_dist_wong,label='method 2', color='green',ls='--')
        #ax.plot(E,Barrier_dist_wong2,label='Method 3', color='blue',ls='--')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('d$^{2}$(E$\sigma$)/$dE^{2}$ (mb MeV$^{-1}$)')
        ax.set_xlim(left=55,right=70)
        ax.set_ylim(top=1500,bottom=0)
        plt.legend()
        plt.grid()
        #plt.savefig('Carrier_dist.png',dpi=1000)
        plt.savefig('Carrier_dist_High.png',dpi=200,bbox_inches='tight',)
        #plt.show()

    if Sim_Analysis == True:
        #First access experimental data
        E_num, T0_num = np.loadtxt('FusProb.dat',unpack=True)
        

        sigma_scan = np.array([1, 2.5, 5, 7.5, 10, 15, 20])
        #sigma_scan = np.array([1,2.5,5,10,15])
        n_sigma = int(len(sigma_scan))
        Energy = np.zeros(shape=[n_sigma,16])
        T_P = np.zeros(shape=[n_sigma,16])
        for X in range(n_sigma):
            #Access simulation scan files
            for i in range(0,16):
                file = pd.read_excel(f'Sigma_{sigma_scan[X]}\E_scan.xlsx',sheet_name=(i+1))
                Energy[X,i] = file.iloc[16,2]
                T_P[X,i] = file.iloc[18,2]
        #COLOR = np.array(['black','Green','Orange','Purple',''])    
        LS =np.array(['dashed', 'dashed', 'dashed', 'solid','dotted','solid','dashed',])
        
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 36})
        plt.rcParams.update({'lines.linewidth' : 4})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        #ax.title.set_text("Transmission probability")
        ax.semilogy(E, T0_E, label='Analytical  - Hill Wheeler',color='blue',ls='-')
        ax.semilogy(E_num, T0_num, ls='--',color='red', label='Numerical - TISE')
        for X in range (n_sigma):
            ax.semilogy(Energy[X,:],T_P[X,:], label=f'$\sigma={sigma_scan[X]}$ fm', marker='x',ls=LS[X],markersize=15)
        ax.set_ylabel('Transmission Probability')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylim(bottom=0.001,top=1)
        ax.set_xlim(left=55,right=70)
        plt.legend()
        plt.grid()
        plt.savefig('Tran_prob_HIGH.png',dpi=300,bbox_inches='tight',)
        #plt.show(
            
    if Sim_Analysis_2 == True:
    #First access experimental data
        E_num, T0_num = np.loadtxt('FusProb.dat',unpack=True)
        

        sigma_scan = np.array([1, 2.5, 5, 7.5, 10, 15, 20])
        #sigma_scan = np.array([1,2.5,5,10,15])
        n_sigma = int(len(sigma_scan))
        Energy = np.zeros(shape=[n_sigma,16])
        T_P = np.zeros(shape=[n_sigma,16])
        for X in range(n_sigma):
            #Access simulation scan files
            for i in range(0,16):
                file = pd.read_excel(f'Sigma_{sigma_scan[X]}\E_scan.xlsx',sheet_name=(i+1))
                Energy[X,i] = file.iloc[16,2]
                T_P[X,i] = file.iloc[18,2]
                
           
        T0_E = np.zeros(len(Energy[0,:]))
        for i in range(len(Energy[0,:])):
            T0_E[i] = 1 / (1+np.exp(-(2*np.pi)*(Energy[0,i]-V_total[V_total_max])/Curv))
        #COLOR = np.array(['black','Green','Orange','Purple',''])    
        LS =np.array(['dashed', 'dashed', 'dashed', 'solid','dotted','solid','dashed',])
        
        plt.figure(figsize=(30,20))
        plt.rcParams.update({'font.size': 36})
        plt.rcParams.update({'lines.linewidth' : 4})
        fig=plt.gcf()
        ax=fig.add_subplot(111)
        for X in range (n_sigma):
            ax.semilogy(Energy[X,:], 100* np.sqrt(((T0_E-T_P[X,:])/T0_E)**2) , label=f'$\sigma={sigma_scan[X]}$ fm', marker='x',ls=LS[X],markersize=15)
        ax.set_ylabel('Relative Difference (%)')
        ax.set_xlabel('Energy (MeV)')
        #ax.set_ylim(bottom=0.001,top=1)
        ax.set_xlim(left=55,right=70)
        plt.legend()
        plt.grid()
        plt.savefig('Prob_FUS_DIFFFFF_HIGH',dpi=300,bbox_inches='tight',)
        #plt.show()


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
    Sigma0 = 7.5 #(fm)
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

    #Plot image for viva powerpoint
    if Show_init_psi == True:
        plt.figure(figsize=(20,20))
        plt.rcParams.update({'font.size': 20})
        plt.rcParams.update({'lines.linewidth' : 2.5})
        fig=plt.gcf()
        ax=fig.add_subplot(211)
        ax.set_ylim(bottom=-0.4,top=0.4)
        ax.set_xlim(left=50,right=90)
        ax.plot(r,np.real(Psi_init)*(np.exp(-(1j*(5.99)*r))), color='blue')
        ax.set_ylabel('$Re[\Psi(r)]$',rotation=0)
        ax.yaxis.set_label_coords(-0.2,0.45)
        ax.set_xlabel('')
        ax.set_xticks([])
        
        ax2=fig.add_subplot(212)
        ax2.set_ylim(bottom=0,top=0.2)
        ax2.set_xlim(left=50,right=90)
        ax2.plot(r,np.abs(Psi_init)**2, color='blue')
        ax2.set_ylabel('$|\Psi(r)|^{2}$',rotation=0)
        ax2.set_xlabel('Radius (fm)')
        ax2.yaxis.set_label_coords(-0.2,0.45)
        
        plt.subplots_adjust(bottom=0.086,right=0.622,top=0.971,left=0.238,hspace=0.1,wspace=0.195)
        plt.show()
        quit()

    #Array of Energies to simulate over
    #Emax = 70
    #Emin = 55
    #n = int(Emax-Emin) + 1    
    #E_SIM = np.linspace(Emin,Emax,n,endpoint=True,dtype=int)
    n=1
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
        
        # total Transmission probability and expectation energy evolution
        T_SIM = np.zeros(nt_save)
        E_expectation = np.zeros(nt_save)
        Sec_Der = np.zeros(nr)
        for j in range(nt_save):
            #Loss of normalisation calculation
            NORM_SIM = np.sum(np.absolute(Psi_FD_Sol[j,:])**2*dr)
            T_SIM[j] = 1 - NORM_SIM
            
            #Energy Expectation value Calculation
            #for i in range(1,nr-1):
            #    Sec_Der[i] = Psi_FD_Sol[j,i+1] - 2 * Psi_FD_Sol[j,i] +Psi_FD_Sol[j,i-1]     
            #E_expectation[j] = np.sum(np.conjugate(Psi_FD_Sol[j,:]) * (( - Sec_Der[:] / (HMC*mu_macron) + Psi_FD_Sol[j,:] * V_total) * dr))

        if Norm_analysis == True:
            plt.figure(figsize=(30,20))
            plt.rcParams.update({'font.size': 36})
            plt.rcParams.update({'lines.linewidth' : 4})
            fig=plt.gcf()
            ax=fig.add_subplot(111)
            ax.plot(t_save,T_SIM, ls='--',color='red')
            ax.text(0.8,0.355,'Max $P_{fus}$ = ' + f'{np.round(np.max(T_SIM),3)}  '+ 
                    '\nFinal $P_{fus}$ = ' + f'{np.round(T_SIM[-1],3)}  ' + 
                    '\nMax difference $P_{fus}$ = ' + f'{np.round(((np.max(T_SIM)-T_SIM[-1])/np.max(T_SIM))*100,2)} %  ',
                    fontsize=36, bbox=dict(facecolor='white',edgecolor='black'))
            ax.set_ylabel('Probability of Fusion')
            ax.set_xlabel('Time ($10^{-22}$ s)')
            ax.set_ylim(bottom=0,top=0.4)
            ax.set_xlim(left=t_min,right=t_max)
            plt.grid()
            #plt.show()
            plt.savefig('Probs_fus_evolution_HIGH',dpi=300,bbox_inches='tight')
            quit()

        
        if FILM_Psi_real == True:
            def init():  
                line.set_data([], []) 
                return line, 
    
            def animate(i):
                y1 = np.real(Psi_FD_Sol[i,:])
                line.set_data(r, y1)
                time = np.round(t_save[i], 1)
                tx.set_text("Gaussian Wavepacket Animation Real \n$t = $"+"{:.1f}".format(time)+' $10^{-22}$')
                print("Psi Real Animation: "+str(np.round(100*(i/(len(t_save))),0))+" %")
                return line, 
    
            ###Animation###
            #plt.rcParams['animation.ffmpeg_path'] = (r"c:\ffmpeg")
            plt.rcParams.update({'font.size': 36})
            plt.rcParams.update({'lines.linewidth' : 4})
            fig, ax = plt.subplots(1,1,figsize=(30,20))
            ax.set_ylim(bottom=-0.4,top=0.4)
            ax.set_xlim(left=1,right=r_max)
            tx = ax.set_title("Gaussian Wavepacket Animation \n$t = 0 $10^{-22}$")
            line, = ax.plot([], [], lw = 1.5)
            ax.set_ylabel('$Re[\Psi(r)]$')
            ax.set_xlabel('Radius (fm)')
            ax2 = ax.twinx()
            ax2.plot(r,V_total,color='green',label='Nuclear Potential')
            ax2.set_ylim(bottom=-150,top=150)
            ax2.set_ylabel('Energy (MeV)')
        
            #30 frames per second
            fps = 30
            # number of frames to be used
            frame_num =  len(t_save)/(t_max*fps)    
            #Total number of frames, 1 second contains 30 frames and reflects 1x10^-22 seconds in sim
            Frames = np.arange(0, len(t_save), frame_num, dtype=int)
            anim = animation.FuncAnimation(fig, animate, init_func = init, frames = Frames, interval = 20, blit = True) 
            FFwriter  = animation.FFMpegWriter(fps=fps)
            anim.save(f'Psi_real_E{E0}.mp4',  writer = FFwriter)
            
        if FILM_Psi_PDF == True:
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
            fps = 30
            # number of frames to be used
            frame_num =  len(t_save)/(t_max*fps)    
            #Total number of frames, 1 second contains 30 frames and reflects 1x10^-22 seconds in sim
            Frames = np.arange(0, len(t_save), frame_num, dtype=int)
            anim = animation.FuncAnimation(fig, animate, init_func = init, frames = Frames, interval = 20, blit = True) 
            FFwriter  = animation.FFMpegWriter(fps=fps)
            anim.save(f'PSi_PDF_E{E0}.mp4',  writer = FFwriter) 


        #Output entire simulation data to Excel sheet
        Params_excel = np.array([
                                ['r_max (fm)',r_max],
                                ['r_min (fm)',r_min],
                                ['dr (fm)',dr],
                                ['# r grid points',nr],
                                ['t_max (10^{-22})',t_max],
                                ['t_min (10^{-22})',t_min],
                                ['dt (10^{-22})',dt],
                                ['# t grid points',nt],
                                ['steps skipped (between saved steps)',nt_skip],
                                ['# saved grid points',nt_save],
                                ['Target Z:',int(144)],
                                ['Projectile Z:',int(16)],
                                ['Reduced Mass (mu)',mu_macron],
                                ['Guassian Width (fm)',Sigma0],
                                ['Guassian Start (fm)',r0],
                                ['Wavepacket Momentum (fm^{-1})',np.round(k0,2)],
                                ['Wavepacket initial Energy (MeV)',E0],
                                ['Final Transmission Probability', np.round(T_SIM[-1],3)],
                                ['Max Transmission Probability', np.round(np.max(T_SIM),3)],
                                ['Final Energy Expectation Energy (MeV)',np.round(E_expectation[-1],3)],
                                ['Max Energy Expectation Energy (MeV)',np.round(np.max(E_expectation),3)],
                                ['Simulation run time (min)',np.round(sim_time/60,0)],
                                ])
        
        data_types = {'value': float}
        df1 = pd.DataFrame(Params_excel,columns=['Parameter','value'])
        #df2 = pd.DataFrame(Psi_FD_Sol,columns=r,index=t_save)

        with pd.ExcelWriter("E_scan.xlsx",mode='a') as writer:
            df1.astype(data_types).to_excel(writer,sheet_name=(f'E-{E0}'))       



    
#V_analysis()
#Main()
