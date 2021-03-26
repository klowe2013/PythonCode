# -*- coding: utf-8 -*-
"""
General functions .py file for analyses. Placed here in order to allow for importing all at once

Created on Wed Aug 12 09:35:00 2020

@author: Kaleb
"""
# Import numpy
import numpy as np
    

# This function gets a PSP kernel for convolution for SDFs
def klGetKern():
    # Set default parameters
    td = 20
    tg = 1
    
    pspTime = list(range(10*td))+[(1)]
    kern = np.multiply((1-np.exp(np.negative(pspTime)/tg)),np.exp(np.negative(pspTime)/td))
    leadingZeros = np.zeros(len(pspTime))
    catKern = np.concatenate((leadingZeros,kern))
    kernOut = catKern/sum(catKern)

    return kernOut


# This function convolves a spike matrix with the PSP kernel from klGetKern to generate an SDF matri
def klGetSDF(spkMat):
    
    # Find minimum spike time overall, shift the spike times, and get the new max and time range
    minSpk = np.nanmin(spkMat)
    spkShift = spkMat-(minSpk-1)
    newMax = np.nanmax(spkShift)
    tRange = [int(np.floor(minSpk)), int(np.floor(minSpk) + np.ceil(newMax))]
    
    # Get convolution kernel
    myKern = klGetKern()*1000

    # Round up each spike time, and place a 1 at each trial (row) and spike time index (column)
    spkInds = np.zeros((np.size(spkMat,axis=0),len(range(np.ceil(newMax).astype(int)))))
    convSpks = np.zeros((np.size(spkMat,axis=0),len(range(np.ceil(newMax).astype(int)))))
    for r in range(np.size(spkMat,axis=0)):
        if sum(~np.isnan(spkShift[r])) > 0:
            theseInds = (np.ceil(spkShift[r,~np.isnan(spkShift[r])])-1)
            spkInds[r,theseInds.astype(int)] = 1
            convSpks[r,] = np.convolve(spkInds[r,],myKern,mode='same')
            rowOver = np.argwhere(convSpks[r,] > 0.1)
            # if len(rowOver) == 0:
            #     convSpks[r,] = np.nan
            # else:
            convSpks[r,:int(rowOver[0])] = np.nan
            convSpks[r,int(rowOver[-1]):(np.size(convSpks,axis=1))] = np.nan
        else:
            spkInds[r,] = np.nan
            convSpks[r,] = np.nan
        
    sdfDict = {'SDF': convSpks, 'Times': np.array(range(tRange[0],tRange[1]))}
    return sdfDict


# Get unique values without nans, from a numpy array
def nunique(myArray):
    # First get unique values
    uVals = np.unique(myArray)
    outVals = uVals[~np.isnan(uVals)]
    return outVals

