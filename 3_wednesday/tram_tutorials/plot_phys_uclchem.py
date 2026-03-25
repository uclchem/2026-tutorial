import numpy as np 
import matplotlib.pyplot as plt
import uclchem
from astropy import constants
from scipy.integrate import simps, quad
from IPython.display import clear_output

def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    if float(a)==1.0:
        return "10^"+str(b)
    else:
        return a+"\\times 10^" + str(b)
    
def live_plot(x, y, figsize=(7,5), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    plt.xlim(0, training_steps)
    plt.ylim(0, 100)
    x= [float(i) for i in x]
    y= [float(i) for i in y]
    
    if len(x) > 1:
        plt.scatter(x,y, label='axis y', color='k') 
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, [x * m for x in x] + b)

    plt.title(title)
    plt.grid(True)
    plt.xlabel('axis x')
    plt.ylabel('axis y')
    plt.show();

species = 'HNCO'#'NH2CHO'#'CH3OH'#'C2H5OH'#'CH3OCH3'#
a_=0.5
Lstar=1e0
zeta=1;zeta_ref=1.0
Tth = 0.
rout=0.01
rflat=0.005
T0=10.
n0=1e8

path_start=f'rout={rout:.3f}pc_rflat={rflat:.3f}pc/starts_norot/'
path_hotcore=f'rout={rout:.3f}pc_rflat={rflat:.3f}pc/hotcores_norot/'

# path_start=f'../grid_folder/IRAS16293/rout={rout:.3f}pc_rflat={rflat:.3f}pc/starts/'
# path_hotcore=f'../grid_folder/IRAS16293/rout={rout:.3f}pc_rflat={rflat:.3f}pc/hotcores/'

# path_start=f'../grid_folder/SgrB2N1/rout={rout:.2f}pc_rflat={rflat:.3f}pc_att/starts/'
# path_hotcore=f'../grid_folder/SgrB2N1/rout={rout:.2f}pc_rflat={rflat:.3f}pc_att/hotcores/'

#pre-stellar phase
starts_df=uclchem.analysis.read_output_file(f"{path_start}{Lstar:.2e}_{T0:.2f}_{n0:.2e}_{a_:.2f}.dat")
# starts_df=uclchem.analysis.read_output_file(f"{path_start}{Lstar:.1e}_{T0:.1f}_{n0:.1e}_{zeta:.1f}_{a_:.1f}.dat")
time_s    = starts_df['Time']
density_s = starts_df['Density']
Av_s      = starts_df['av']
temp_s    = starts_df['gasTemp']
abund_s   = starts_df[species]#['CH3OCH3']#['NH2CHO']#["CH3OH"]#['C2H5OH']#

hotcore_df=uclchem.analysis.read_output_file(f"{path_hotcore}{Lstar:.2e}_{T0:.2f}_{n0:.2e}_{a_:.2f}.dat")
# hotcore_df=uclchem.analysis.read_output_file(f"{path_hotcore}{Lstar:.1e}_{T0:.1f}_{n0:.1e}_{zeta:.1f}_{a_:.1f}.dat")
time_hc    = hotcore_df['Time']#np.ones_like(time_s)*np.nan#
density_hc = hotcore_df['Density']#np.ones_like(time_s)*np.nan#
Av_hc      = hotcore_df['av']#np.ones_like(time_s)*np.nan#
temp_hc    = hotcore_df['gasTemp']#np.ones_like(time_s)*np.nan#
abund_hc   = hotcore_df[species]#np.ones_like(time_s)*np.nan##['CH3OCH3']#['NH2CHO']#["CH3OH"]#['C2H5OH']#
# try:
#     hotcore_df1=uclchem.analysis.read_output_file(f"../grid_folder_test/SgrB2N1_rin=0.02pc/rout=0.3pc_a={a_:.2f}/hotcores_norot/{Lstar:.1e}_10.00_1.00e+07_{a_:.2f}.dat")
#     time_hc1    = hotcore_df1['Time']
#     abund_hc1   = hotcore_df1[species]#['CH3OCH3']#['NH2CHO']#["CH3OH"]#['C2H5OH']#
# except:
#     time_hc1=np.ones(20)
#     abund_hc1=np.ones(20)

n_models = 100
fig,ax = plt.subplots(figsize=(10,8))
for i in np.arange(0,100+4,4):#range(n_models):
    t1=time_s[i+1::n_models]; t2=time_hc[i+1::n_models]
    n1=density_s[i+1::n_models]; n2=density_hc[i+1::n_models]
    if i == 48:
        ax.semilogy(t1/1e6-t1.max()/1e6,n1,'k-')
        ax.semilogy(t2/1e6,n2,'k-')
    else:
        ax.semilogy(t1/1e6-t1.max()/1e6,n1,'k-')
        ax.semilogy(t2/1e6,n2,'k-')
ax.axvline(x=0,ls='--',color='k')
ax.text(0.25,1.03,'pre-stellar stage',horizontalalignment='center',transform=ax.transAxes)
ax.text(0.75,1.03,'heating stage',horizontalalignment='center',transform=ax.transAxes)
# ax.set(xscale='symlog',yscale='symlog')#,ylim=(30,5e8))
# ax.set(yscale='symlog')#,ylim=(30,5e8))
ax.set_xlabel('$\\sf Time\,(Myr)$')
ax.set_ylabel('$\\sf Density\,(cm^{-3})$')

#plot the mass over the time
fig,ax = plt.subplots(figsize=(10,8))
rmin = rout/(n_models)
r = np.linspace(rmin,rout,n_models)[::-1]
for i in np.arange(0,100+4,4):#range(n_models):
    t1=time_s[i+1::n_models]; t2=time_hc[i+1::n_models]
    n1=density_s[i+1::n_models]; n2=density_hc[i+1::n_models]
    t1_array = t1/1e6-t1.max()/1e6
    t2_array = t2/1e6
    t_array = np.concatenate((t1_array,t2_array))
    nH_array= np.concatenate((n1,n2))
    if i == 0 or i == n_models:
        continue
    
    # elif i==n_models-4:
    #     rrec  = r[i] * constants.pc.cgs.value
    #     mass_i = 4/3 * np.pi * pow(rrec,3) * nH_array * 2.8 * constants.m_p.cgs.value/constants.M_sun.cgs.value
    #     ax.semilogy(t_array,mass_i,'g-')        
    #     ax.semilogy(t_array,nH_array,'g--')

    else:
        rrec  = r[i] * constants.pc.cgs.value
        rprev = r[i-1] * constants.pc.cgs.value
        mass_i = 4/3 * np.pi * (pow(rprev,3) - pow(rrec,3)) * nH_array * 2.8 * constants.m_p.cgs.value/constants.M_sun.cgs.value
        if i==4:
            ax.semilogy(t_array,mass_i,'-', color='magenta',label='Cell \#4')
        elif i==52:
            ax.semilogy(t_array,mass_i,'-', color='cyan', label='Cell \#52')
        elif i==96:
            ax.semilogy(t_array,mass_i,'-', color='darkorange', label='Cell \#96')
        else:
            ax.semilogy(t_array,mass_i,'-',color='gray',alpha=0.5)
        # ax.semilogy(t_array,nH_array,'k--')


 

    # # elif i==n_models:
    # #     
    # else:
    #     # rprev = r[i-1] * constants.pc.cgs.value
    #     # rrec  = r[i]   * constants.pc.cgs.value

    #     # mass_i = 4/3 * np.pi * (pow(rprev,3) - pow(rrec,3)) * nH_array * 2.8 * constants.m_p.cgs.value/constants.M_sun.cgs.value
    #     # ax.semilogy(t_array,mass_i,'k-')
    #     # # ax.semilogy(t_array,nH_array)
    #     continue

ax.axvline(x=0,ls='--',color='k')
ax.text(0.25,1.03,'pre-stellar stage',horizontalalignment='center',transform=ax.transAxes)
ax.text(0.75,1.03,'heating stage',horizontalalignment='center',transform=ax.transAxes)
ax.set_xlabel('$\\sf Time\,(Myr)$')
ax.set_ylabel('$\\sf Mass\,(M_{\odot})$')
ax.legend()


fig,ax = plt.subplots(figsize=(10,8))
for i in np.arange(0,100+4,4):#range(n_models):
    t1=time_s[i+1::n_models]; t2=time_hc[i+1::n_models]
    av1=Av_s[i+1::n_models]; av2=Av_hc[i+1::n_models]
    ax.semilogy(t1/1e6-t1.max()/1e6,av1,'k-')
    ax.semilogy(t2/1e6,av2,'k-')
ax.axvline(x=0,ls='--',color='k')
ax.text(0.25,1.03,'pre-stellar stage',horizontalalignment='center',transform=ax.transAxes)
ax.text(0.75,1.03,'heating stage',horizontalalignment='center',transform=ax.transAxes)
# ax.set(xscale='symlog',yscale='symlog')#,ylim=(30,5e8))
# ax.set(yscale='symlog')#,ylim=(30,5e8))
ax.set_xlabel('$\\sf Time\,(Myr)$')
ax.set_ylabel('$\\sf A_{V}\,(mag.)$')

fig,ax = plt.subplots(figsize=(10,8))
for i in range(n_models):
    t1=time_s[i+1::n_models]; t2=time_hc[i+1::n_models]
    T1=temp_s[i+1::n_models]; T2=temp_hc[i+1::n_models]
    ax.plot(t1-t1.max(),T1,'k-')
    ax.plot(t2,T2,'k-')
ax.axvline(x=0,ls='--',color='k')
ax.text(0.17,0.5,'pre-stellar stage',horizontalalignment='center',transform=ax.transAxes,rotation=-270)
ax.text(0.25,0.5,'heating stage',horizontalalignment='center',transform=ax.transAxes,rotation=-270)
# ax.set(xscale='symlog',yscale='symlog')#,ylim=(30,5e8))
ax.set(xscale='symlog')#,ylim=(30,5e8))
ax.set_xlabel('$\\sf Time\,(yr)$')
ax.set_ylabel('$\\sf Temperature\,(K)$')

##pre-phase
fig,ax = plt.subplots(figsize=(10,8))
for i in range(n_models):
    t1=time_s[i+1::n_models]
    abund1=abund_s[i+1::n_models]
    ax.loglog(t1,abund1)

##hot-core
fig,ax = plt.subplots(figsize=(10,8))
# Track with labels have already been used
plotted_labels = set()

for i in range(n_models):
    t1=time_s[i+1::n_models]
    t2=time_hc[i+1::n_models]
    abund1=abund_s[i+1::n_models]
    abund2=abund_hc[i+1::n_models]#;abund21=abund_hc1[i+1::n_models]
    
    print('abund1=',np.array(abund1)[-1],'abund2=',np.array(abund2)[0])
    
    max_temp = temp_hc[i+1::n_models].max()
    print('imod=%d, T=%.0f'%(i,max_temp))
    if max_temp>=150.:
        label='$\\sf T\geq 150\,K$'
        color='red'
    
    elif max_temp>=100.:
        label='$\\sf 100\leq T<150\,K$'
        color='darkorange'

    elif max_temp>=50.:
        label='$\\sf 50\leq T<100\,K$'
        color='navy'
    
    else:
        label='$\\sf T<50\,K$'
        color='cyan'

    if label not in plotted_labels:
        ax.loglog(t2, abund2, ls='-', color=color, label=label)
        plotted_labels.add(label)
    else:
        ax.loglog(t2, abund2, ls='-', color=color)

# ax.set(xscale='symlog',yscale='symlog')#,ylim=(30,5e8))
# ax.set(xscale='symlog')#,ylim=(30,5e8))
ax.set_xlabel('$\\sf Time\,(yr)$')
ax.set_ylabel('$\\sf Abundance$')
ax.legend(fontsize=20)

# ###                                                                 ###
# ###     PLOT PHYSICAL PROFILES vs. TIME for prestellar case         ###
# ###                                                                 ###
# # Define a function for extrapolation
# def extrapolate(x_vals, slope, intercept):
#     return slope * x_vals + intercept

# rmin = rout/(n_models)
# r = np.linspace(rmin,rout,n_models)[::-1]#[1:]#*constants.pc.cgs.value 
# r_new = np.linspace(rmin,100*rout,100*n_models)#[1:]#*constants.pc.cgs.value 
# fig,ax = plt.subplots(figsize=(12,8))
# for i in [0,50,100,105,106,107,108,109, 110,115,120,130,140,150,160]:
#     density_overtime = np.array(density_hc[1+100*i:100+1+100*i])

#     # Power-law function with fixed exponent and x0
#     def model(x):
#         x0 = 0.05  # known
#         return density_overtime.max()/(1. + (x/x0)**(2.4))


#         # if x<=x0:
#         #     return density_overtime.max()
#         # else:
#         #     return density_overtime.max() * (x / x0) ** (-2.4)

#     # density_new = model(r_new)#[model(r_) for r_ in r_new]
#     # N_rout=density_overtime.max()*0.05*constants.pc.cgs.value/(2.4-1) * (r.max()/0.05)**(1-2.4)
#     # print('N=',N_rout,'Av=',N_rout/5.8e21*4)
#     ax.loglog(r,density_overtime)
#     # ax.loglog(r_new,density_new,'--')
#     # ax.loglog(r,density_s[1+100:100+1+100])
#     # ax.loglog(r,density_s[1+100+100:100+1+100+100])

# ###                                                                 ###
# ###             PLOT PHYSICAL PROFILES vs. DISTANCE                 ###
# ###                                                                 ###

# rmin = rout/(n_models)
# r = np.linspace(rmin,rout,n_models)[::-1]#[1:]#*constants.pc.cgs.value 
# fig,ax = plt.subplots(figsize=(12,8))
# Tdust_r = np.zeros(n_models)
# ngas_r = np.zeros(n_models)
# for i in range(n_models):
#     Tdust_r[i] = np.array(temp_hc[i+1::n_models])[-1]
#     ngas_r[i] = np.array(density_hc[i+1::n_models])[-1]


##-------------------------------------------------------------------------------------------------##
##-----------------------------------radius vs. CRIR-----------------------------------------------##
##-------------------------------------------------------------------------------------------------##
## For SgrB2(N1)
fig,ax = plt.subplots(figsize=(10,8))
regions=['West','South']
linestyles=['-','-.']
rout  = 0.3
Lstar = 6e6
av0   = 2
Tinit = 10
nH    = 1e7
a     = 0.4

for i,region in enumerate(regions):
    if region == 'South':
        rflat = 0.012
        zeta_scale  = 20
    elif region == 'West':
        rflat= 0.02
        zeta_scale = 30
    else:
        IOError('No region is found!')

    outputFile     = f"../grid_folder/SgrB2N1/rout={rout:.2f}pc_rflat={rflat:.3f}pc_att/hotcores_av0={av0}/{Lstar:.1e}_{Tinit:.1f}_{nH:.1e}_{zeta_scale:.1f}_{a:.1f}.dat"
    data_uclchem = uclchem.analysis.read_output_file(outputFile)
    models_number = 30
    r_model = np.linspace(rout,rout/models_number,models_number)

    zeta = data_uclchem['zeta']
    zeta_r=np.zeros(models_number)
    for j in range(models_number):
        zeta_r[j] = np.array(zeta[j+1::models_number])[-1]          

    ax.plot(r_model,zeta_r,'k',ls=linestyles[i], label='SgrB2(N1)-'+region)
ax.set_xlabel('Distance (pc)')
ax.set_ylabel('Cosmic-ray ionisation rate ($\\sf \\zeta \\times 1.3\\times 10^{-17}\,s^{-1}$)')
ax.legend()
plt.show()
