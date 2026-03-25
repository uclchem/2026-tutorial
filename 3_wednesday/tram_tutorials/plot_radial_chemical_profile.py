import numpy as np
import matplotlib.pyplot as plt
import uclchem
from astropy import constants
from collections import OrderedDict

ls                  = OrderedDict(
                                 [
                                  ('solid',               (0, ())),
                                  ('dashed',              (0, (5, 5))),
                                  # ('solid',               (0, ())),
                                  ('dashdotted',          (0, (5, 4, 1, 6))),
                                  ('dotted',              (0, (1, 5))),

                                  ('loosely dashed',      (0, (5, 15))),
                                  ('densely dashed',      (0, (5, 1))),
                                  ('loosely dotted',      (0, (1, 10))),
                                  ('densely dotted',      (0, (1, 1))), 
                                  ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),

                                  ('loosely dashdotted',  (0, (3, 10, 1, 10))),
                                  ('densely dashdotted',  (0, (3, 1, 1, 1))),

                                  ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                                  ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
keys = list(ls.keys())

def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    if float(a)==1.0:
        return "10^"+str(b)
    else:
        return a+"\\times 10^" + str(b)

def underscore_numbers(text):
    # Iterate over each character in the text
    result = ''.join(f"$_{char}$" if char.isdigit() else char for char in text)
    return result

au = constants.au.cgs.value
pc = constants.pc.cgs.value
def pc_to_au(pc_value):
	return pc_value*pc/au

def au_to_pc(au_value):
	return au_value*au/pc

colors=['dodgerblue','darkorange','turquoise',"navy",'magenta','purple','darkseagreen','violet']
# markers=['d','s','o','p','>']

#-------------------------------------------------------------------------------------------------##
#---------------------------------X vs. distance--------------------------------------------------##
#-------------------------------------------------------------------------------------------------##
a_=0.5
# Lstar=1e5
T0=10.0
models_number=100
species=["CH3OH","CH3OCH3","C2H5OH","NH2CHO","CH3CHO","CH2CO","CH3CHOH", "HCOOH"]#,"NH2CHO","HCOOH","CH3OCH3","C2H5OH"]
Lstar_range=[1e0,1e1,1e2,1e3,1e4,1e5,1e6];rout_range=[0.01,0.015,0.03,0.06,0.1,0.2,0.5]
rflat_range=[0.005,0.005,0.005,0.03,0.05,0.05,0.05]
# Lstar_range=[1e0,1e1,1e2,1e3];rout_range=[0.03,0.03,0.04,0.05]

fig,axs=plt.subplots(4,2,figsize=(14,14),sharex=False,sharey=True)
fig.subplots_adjust(top=0.95,bottom=0.07,wspace=0.06,hspace=0.35)

# Keep track of plotted subplots
plotted_axes = set()

for idx,Lstar in enumerate(Lstar_range):
    # path=f'../grid_folder/rout={rout_range[idx]:.3f}pc_a={a_:.2f}/hotcores_norot/'
    path=f'rout={rout_range[idx]:.3f}pc_rflat={rflat_range[idx]:.3f}pc/hotcores_norot/'
    if Lstar in [1e0,1e1,1.e2]:
        n0=1e8
        tstop=2e5
    elif Lstar in [1e3]:
        n0=1e7
        tstop=2.e5
    elif Lstar in [1e6]:
        n0=1e7
        tstop=2e4
    else:
        n0=1e7
        tstop=5e4

    outputFile=f"{path}{Lstar:.2e}_{T0:.2f}_{n0:.2e}_{a_:.2f}.dat"
    data_uclchem = uclchem.analysis.read_output_file(outputFile)
    time = data_uclchem['Time'][1::models_number]
    r_model = np.linspace(rout_range[idx]/models_number,rout_range[idx],models_number)[::-1]

    row=idx//2
    col=idx%2
    for i,specie in enumerate(species):
        x=[]
        for j in range(models_number):
            x_array = data_uclchem[specie][j+1::models_number]
            xi = np.array(x_array)[np.argmin(abs(time-tstop))]
            if Lstar==1.e3:
                if j==models_number-1:
                    xi=np.nan
            x.append(xi)
        
        ## plot data if there's something to plot
        if any(x):
            axs[row,col].loglog(r_model[1:],x[1:],color=colors[i],ls=ls[keys[i]],label=underscore_numbers(specie))#label='$L_{\\ast}=%s$'%sci_notation(Lstar,1))
            # axs.loglog(r_model,x,'-',label=underscore_numbers(specie))
            plotted_axes.add((row,col))
    axs[row,col].text(0.55,0.82,'$L_{\\ast}=%s L_{\odot}$'%(sci_notation(Lstar,1)),transform=axs[row,col].transAxes,ha='left')
    axs[row,col].text(0.55,0.7,'$t=%s\,\\sf yr$'%(sci_notation(tstop,1)),transform=axs[row,col].transAxes,ha='left')

    secax = axs[row,col].secondary_xaxis('top', functions=(pc_to_au,au_to_pc))
    secax.set_xscale('log')
    axs[row,col].tick_params(axis='x', which='both', top=False, labeltop=False)
    secax.tick_params(axis='x', which='both', top=True)
    if row==0:
        secax.set_xlabel('$\\sf Distance\\,(au)$',labelpad=10)
    
for row in range(np.shape(axs)[0]):
    for col in range(np.shape(axs)[1]):
        if (row,col) not in plotted_axes:
            fig.delaxes(axs[row,col]) # Remove empty subplot

axs[2,1].legend(loc='lower right',ncols=2,bbox_to_anchor=(1.0,-1.2),fontsize=18)
# axs[0,0].set_xlim([2e-4,0.2])
axs[0,0].set_ylim([9e-13,1e-4])
# [axs[1,i].set_xlabel('Distance (pc)') for i in [0,1]]
axs[np.shape(axs)[0]-1,0].set_xlabel('Distance (pc)')
[axs[i,0].set_ylabel('Abundance') for i in range(np.shape(axs)[0])]
[axs[i,1].tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            labelleft=False, # labels along the bottom edge are off)
            labelright=True,
        )
        for i in range(np.shape(axs)[0])]

[axs[i,1].set_ylabel('Abundance') for i in range(np.shape(axs)[0])]
[axs[i,1].yaxis.set_label_position('right') for i in range(np.shape(axs)[0])]
plt.show()