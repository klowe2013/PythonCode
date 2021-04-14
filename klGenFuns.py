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


def PoissonSpikeMat(in_sdf, n_trs = 100, res = 5, offset = -500):
    import numpy as np
    
    # Set up a spiking lambda function
    poiss_rate = lambda perc_point, lamb: np.log(1-perc_point)/(-1*lamb)
    
    # Set up outputs/midpoints
    n_res = in_sdf.shape[0]//res
    my_lambs = np.empty([n_res,1])*np.nan
    tr_spks = [np.array([]) for i in range(n_trs)]
    
    for it in range(n_trs):
        # Loop over in_sdf by resolution to get mean firing rates
        for ir in range(n_res):
            # Isolate this time bin
            my_times = np.arange(0,res)+(res*ir)
            min_t = min(my_times)
            
            # Get mean value (lambda) for Poisson sampler
            my_lambs[ir] = np.mean(in_sdf[my_times])
            
            # Set up counters
            this_section_itis = [0.]
            this_section_times = [0.]
            
            if not np.isnan(my_lambs[ir]):
                # Do a while loop to sample until the end of the time bin
                while this_section_times[-1] < res:
                    # Get a random value between 0 and 1
                    rand_val = np.random.uniform()
                    next_iti = poiss_rate(rand_val, my_lambs[ir])
                    this_section_itis = np.append(this_section_itis,next_iti*1000)
                    this_section_times = np.append(this_section_times,sum(this_section_itis))
                # Cut out spikes that went past the bin
                while this_section_times[-1] > res:
                    this_section_times = this_section_times[:-1]
            if len(this_section_times) > 1:
                tr_spks[it] = np.append(tr_spks[it],np.round(this_section_times[1:] + (res/2) + min_t))
        # End IR loop
    # End IT loop
    
    n_spks = [len(tr_spks[i]) for i in range(len(tr_spks))]
    all_spks = np.empty([n_trs,max(n_spks)])*np.nan
    for it in range(n_trs):
        all_spks[it,0:n_spks[it]] = tr_spks[it] + offset
    
    return all_spks