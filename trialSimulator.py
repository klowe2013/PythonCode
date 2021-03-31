# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:27:22 2021

@author: klowe
"""

# Do imports
import numpy as np
from SimUtils import *
import matplotlib.pyplot as plt
import seaborn as sns
    
types = ['hh','hl','lh','ll']
N_TRS = 50
TSTEP = 2

rts = [[],[],[],[]]
unit_activities = [[],[],[],[]]

for i in range(4):
    for it in range(N_TRS):
        if it % 10 == 0:
            print('Working on ' + types[i] + ' trial ' + str(it))
        unit_activity, rt = SimTrialLoop(types[i])
        rts[i].append(rt)
        unit_activities[i].append(unit_activity)
        

# Get CDFs of RTs by getting the min and max of all conditions
min_val_all = min([min(rts[i]) for i in range(len(rts))])
max_val_all = max([max(rts[i]) for i in range(len(rts))])

# Pull the bins and CDFs
cdf_calcs = [GetCDF(np.array(rts[i]), min_val=min_val_all, max_val=max_val_all) for i in range(len(rts))]
bins = [cdf_calcs[i][1] for i in range(len(rts))]
cdfs = [cdf_calcs[i][0] for i in range(len(rts))]

# Plot them
clr_vals = [[.2,.2,.8],[0,0,.5],[.8,.2,.2],[.5,0,0]]
plt.figure()
for i in range(len(rts)):
    plt.plot(bins[i],cdfs[i],color=clr_vals[i])
plt.show()

# Let's calculate the SIC function to assess architecture. Get S(t) = 1-F(t) and calculate SIC(t)
sf_array = [1-np.array(cdfs[i]) for i in range(len(rts))]
sic = (sf_array[0]-sf_array[1]) - (sf_array[2]-sf_array[3])

# For each condition, pull singleton and mov unit activities
in_rf_s = [[],[],[],[]]
out_rf_s = [[],[],[],[]]
in_rf_m = [[],[],[],[]]
out_rf_m = [[],[],[],[]]
for i in range(len(unit_activities)):
    sing_activities = [unit_activities[i][ii]['sing'] for ii in range(len(unit_activities[i]))]
    in_rf_s[i] = np.array([sing_activities[ii][:,0] for ii in range(len(sing_activities))])
    out_rf_s[i] = np.array([sing_activities[ii][:,4] for ii in range(len(sing_activities))])
    
    mov_activities = [unit_activities[i][ii]['mov'] for ii in range(len(unit_activities[i]))]
    in_rf_m[i] = np.array([mov_activities[ii][:,0] for ii in range(len(sing_activities))])
    out_rf_m[i] = np.array([mov_activities[ii][:,4] for ii in range(len(sing_activities))])


# Now, let's plot singleton unit activity    
plt.figure()
for i in range(len(in_rf_s)):
    plt.plot(np.nanmean(in_rf_s[i],axis=0),color=clr_vals[i],linewidth=3)
    plt.plot(np.nanmean(out_rf_s[i],axis=0),color=clr_vals[i],linewidth=1)
plt.ylim([0,200])
plt.show()

# And plot mov unit activity
plt.figure()
for i in range(len(in_rf_m)):
    plt.plot(np.nanmean(in_rf_m[i],axis=0),color=clr_vals[i],linewidth=3)
    plt.plot(np.nanmean(out_rf_m[i],axis=0),color=clr_vals[i],linewidth=1)
plt.ylim([0,200])
plt.show()