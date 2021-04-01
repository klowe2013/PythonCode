# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 08:52:02 2020

@author: Kaleb
"""

from scipy.io import loadmat
import h5py
import numpy as np
import matplotlib.pyplot as plt
import klGenFuns as klf
import matplotlib
from matplotlib.path import Path

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Set up windows and defaults
vWind = np.array(range(175,275))

f = h5py.File('./testSpikeMat.mat', 'r')
arrays = {}
for k, v in f.items():
    print(k)
    print(v)
    arrays[k] = np.array(v)
f.close()
    
spkMat = arrays['spkMat']

vDict = klf.klGetSDF(spkMat)
vTimes = vDict['Times']
vSDF = vDict['SDF']

# Now load in the Task struct
f = h5py.File('./testTask.mat', 'r')
taskArrays = {}
for k, v in f.items():
    taskArrays[k] = np.array(v)
f.close()

# Get unique target locations
uLocs = klf.nunique(taskArrays['TargetLoc'])
uLocs = uLocs[uLocs % 45 == 0]

# Set up plotting stuff
cm = matplotlib.cm.get_cmap('jet')
myMap = cm(np.linspace(0,1,len(uLocs)))

# Plot correct responses for each location
meanVals = np.zeros(len(uLocs))
for i in range(len(uLocs)):
    #myCrit = np.all((taskArrays['TargetLoc']==uLocs[i],taskArrays['Correct']==1, taskArrays['SingletonDiff'] > 0.0), axis=0)
    myCrit = (taskArrays['TargetLoc']==uLocs[i]) & (taskArrays['Correct']==1) & (taskArrays['SingletonDiff'] > 0.0)
    thisLocSDF = vSDF[myCrit[0],]
    plt.plot(vTimes,np.nanmean(thisLocSDF,axis=0),color=myMap[i])
    meanVals[i] = np.nanmean(np.nanmean(thisLocSDF[:,np.in1d(vTimes,vWind)],axis=1),axis=0)
    
plt.xlim([-100,400])

ax = plt.gca()
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(20))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))

yl = plt.ylim()

verts = [(vWind[0], yl[0]), (vWind[0], yl[1]), (vWind[-1], yl[1]), (vWind[-1], yl[0]), (vWind[0], yl[0])]
codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
path = Path(verts,codes)
patch = matplotlib.patches.PathPatch(path, facecolor = (0., 0., 0., 0.5))
ax.add_patch(patch)

plt.savefig('./testSpikeSDFs.pdf',format='pdf', transparent=True)

plt.figure(2)
plt.polar(np.deg2rad(np.append(uLocs,[uLocs[0]])),np.append(meanVals,[meanVals[0]]))
for i in range(len(uLocs)):
    plt.polar([0,np.deg2rad(uLocs[i])],[0,meanVals[i]],color=myMap[i])
    
plt.savefig('./testSpikePolar.pdf',format='pdf', transparent=True)
