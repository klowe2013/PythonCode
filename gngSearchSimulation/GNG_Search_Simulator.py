# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 07:33:56 2021

@author: klowe
"""
# Do imports
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os
from multiprocessing import Pool

def main():
    # Parameter setup
    types = ['hh','hl','lh','ll','h0','l0']
    N_TRS = 5
    TSTEP = 1
    PARALLELIZE = True
    CORE_PROP = .75
    DO_SAVE = True
    VERBOSE = True
    PAR_METHOD = 'pool' # 'pool' to use multiprocessing, 'spark' to use PySpark
    
    # Initialize Spark context, if desired
    if PARALLELIZE and PAR_METHOD == 'spark':
        try:
            sc = GetSparkContext(core_prop = CORE_PROP)
        except:
            print('Can\'t initialize Spark Context. Reverting to \'Pool\' method')
            PAR_METHOD = 'pool'
        
    # Initialize condition-wise outputs
    rts = [[] for i in range(len(types))]
    unit_activities = [[] for i in range(len(types))]
    
    start = perf_counter()
    # Start condition loop
    for i in range(len(types)):
        # If we want to parallelize this with PySpark, use a map/reduce approach
        if PARALLELIZE:
            if VERBOSE:
                print('Working on ' + types[i] + ' in parallel context')
            if PAR_METHOD == 'pool':
                # Get number of cores
                try:
                    n_cores = len(os.sched_getaffinity(0))
                except:
                    import psutil
                    n_cores = psutil.cpu_count()
                with Pool(int(n_cores*CORE_PROP)) as p:
                    par_vals = p.map(PoolHelper, [[types[i],TSTEP,t] for t in range(N_TRS)])
            elif PAR_METHOD == 'spark':
                par_vals = sc.parallelize(range(N_TRS)).map(lambda x: SimTrialLoop(types[i], tstep=TSTEP, r_state_in=x)).collect()
            rts[i] = [par_vals[ii][1] for ii in range(N_TRS)]
            unit_activities[i] = [par_vals[ii][0] for ii in range(N_TRS)]
        else:
            # Start trial loop
            for it in range(N_TRS):
                if it % 10 == 0:
                    print('Working on ' + types[i] + ' trial ' + str(it))
                unit_activity, rt, r_val = SimTrialLoop(types[i], tstep=TSTEP, r_state_in=it)
                rts[i].append(rt)
                unit_activities[i].append(unit_activity)
    end = perf_counter()
    
    if VERBOSE:
        print('%d trial types completed in %.3f seconds' % (len(types), start-end))
    
    # Get CDFs of RTs by getting the min and max of all conditions
    min_val_all = min([min(rts[i]) for i in range(len(rts))])
    max_val_all = max([max(rts[i]) for i in range(len(rts))])
    
    # Pull the bins and CDFs
    cdf_calcs = [GetCDF(np.array(rts[i]), min_val=min_val_all, max_val=max_val_all) for i in range(len(rts))]
    bins = [cdf_calcs[i][1] for i in range(len(rts))]
    cdfs = [cdf_calcs[i][0] for i in range(len(rts))]
    
    # Plot them
    clr_vals = [[.2,.2,.8],[0,0,.5],[.8,.2,.2],[.5,0,0],[.2,.2, .8],[.8,.2,.2]]
    line_styles = ['solid','solid','solid','solid','dashed','dashed']
    plt.figure()
    for i in range(len(rts)):
        plt.plot(bins[i],cdfs[i],color=clr_vals[i])
    plt.show()
    
    # Let's calculate the SIC function to assess architecture. Get S(t) = 1-F(t) and calculate SIC(t)
    sf_array = [1-np.array(cdfs[i]) for i in range(len(rts))]
    sic = (sf_array[0]-sf_array[1]) - (sf_array[2]-sf_array[3])
    plt.figure()
    plt.plot(bins[0],sic)
    
    # For each condition, pull singleton and mov unit activities
    in_rf_s = [[] for i in range(len(types))]
    out_rf_s = [[] for i in range(len(types))]
    in_rf_m = [[] for i in range(len(types))]
    out_rf_m = [[] for i in range(len(types))]
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
        plt.plot(np.nanmean(in_rf_s[i],axis=0),color=clr_vals[i],linewidth=3,linestyle=line_styles[i])
        plt.plot(np.nanmean(out_rf_s[i],axis=0),color=clr_vals[i],linewidth=1,linestyle=line_styles[i])
    plt.ylim([0,200])
    plt.show()
    
    # And plot mov unit activity
    plt.figure()
    for i in range(len(in_rf_m)):
        plt.plot(np.nanmean(in_rf_m[i],axis=0),color=clr_vals[i],linewidth=3)
        plt.plot(np.nanmean(out_rf_m[i],axis=0),color=clr_vals[i],linewidth=1)
    plt.ylim([0,200])
    plt.show()

    if DO_SAVE:
        import pickle
        save_obj = {'Units': unit_activities, 'RTs': rts, 'Types': types}
        save_file = open('./testSave.obj','wb')
        pickle.dump(save_obj, save_file)


def PoolHelper(in_list):
    unit_activities, rt, r_val = SimTrialLoop(in_list[0], tstep=in_list[1], r_state_in=in_list[2])
    return unit_activities, rt


def SetupParams():
    # Set up parameters for the model simulations. All as a dict for readability
    params = {}
    
    # Start with parameters for modulating intermediate "color as singleton" units
    params['same_drive'] = .01
    params['other_drive'] = 0.0
    params['same_inhib'] = .001
    params['lat_colors'] = 0.0
    
    # Params for modulating singleton units
    params['sing_drive'] = 0.001
    params['sing_lateral'] = 0.002
    params['ng_vis_inhib'] = 0.02
    params['go_vis_facil'] = 0.0
    params['v_go_gate'] = 50
    params['singk'] = 0.001
    
    # Params for modulating CDT intermediate neurons
    params['ar_drive'] = 0.05
    params['ar_lateral'] = 0.005
    params['ng_width'] = 0.15
    params['go_center'] = 3
    params['go_shape'] = 2
    
    # Params for modulating GO/NO-GO units
    params['b_go'] = 0.006
    params['b_stop'] = 0.006
    params['stim_go'] = .002
    params['stim_ng'] = 0.002
    params['gngk'] = 0.005
    
    # Params for modulating mov neurons
    mu_vals = [0., 0., 0., 0.]
    beta_vals = [.048, .038, .024, .024]
    params['vm_drive'] = 0.08
    params['go_drive'] = 0.08
    params['ff_inhib'] = [0.0, mu_vals[0], mu_vals[1], mu_vals[2], mu_vals[3], mu_vals[2], mu_vals[1], mu_vals[0]]
    params['lat_inhib'] = [0.0, beta_vals[0], beta_vals[1], beta_vals[2], beta_vals[3], beta_vals[2], beta_vals[1], beta_vals[0]]
    params['gate'] = 50
    params['m_go_gate'] = 150
    params['m_ng_gate'] = 20
    params['movk'] = 0.005
    params['go_mdrive'] = 0.0
    params['ng_mdrive'] = 0.2
    
    # General parameters
    params['k'] = .0005
    
    return params


def SimTrialLoop(trial_type, tstep=5, r_state_in=None):

    # Do imports
    import numpy as np
    import random
    #from SimUtils import *
    
    # Set constants/Defaults
    NOISE_SD = 0.05
    SING_NOISE_SD = 0.5
    CLR_STD = 5
    AR_STD = 5
    CLR_STEP = 10
    AR_STEP = .05
    RED_SING = True
    MOV_THRESH = 100
    TR_DUR = 600
    NOISE_OFF = False
    SET_SIZE = 8
    BL_FIRING = 10
    
    BL_START = -300
    LAR_VALS = np.arange(-1,1+AR_STEP,AR_STEP)
    CLR_VALS = np.arange(0,360,CLR_STEP)
    
    # Set random seed
    if r_state_in is not None:
        random.seed(r_state_in)
        np.random.seed(r_state_in)
        
    if NOISE_OFF:
        NOISE_SD = 0
        SING_NOISE_SD = 0
        CLR_STD = 0
        AR_STD = 0
    
    # Decode the trial type
    is_hi, is_hd, is_go = DecodeTrialType(trial_type)
    
    # Make arrays for stim_colors
    if is_hi:
        if RED_SING:
            stim_colors = np.ones(SET_SIZE)
            stim_colors[0] = 0
        else:
            stim_colors = np.ones(SET_SIZE)*0
            stim_colors[0] = 1
    else:
        if RED_SING:
            stim_colors = np.ones(SET_SIZE)*2
            stim_colors[0] = 0
        else:
            stim_colors = np.ones(SET_SIZE)*3
            stim_colors[0] = 1
    
    if not is_go:
        stim_elongs = np.ones(SET_SIZE)*0
    elif is_hd:
        stim_elongs = np.ones(SET_SIZE)*2
    else:
        stim_elongs = np.ones(SET_SIZE)*1
    
    # Set up parameter set
    p = SetupParams()
        
    # Now, set up color transients
    clr_transients = SetupClrTransients(stim_colors.astype(int), tr_duration=TR_DUR, clr_step=CLR_STEP, clr_std=CLR_STD, bl_start=BL_START)
    ar_transients = SetupShapeTransients(stim_elongs.astype(int), tr_duration=TR_DUR, ar_step=AR_STEP, ar_std=AR_STD, bl_start=BL_START)
    transient_times = np.arange(BL_START,TR_DUR)
    
    # Cut down to just t > 0
    sub_clr_resp = [*map(lambda x: x[transient_times >= 0,:], clr_transients)]
    sub_ar_resp = [*map(lambda x: x[transient_times >= 0,:], ar_transients)]
    sub_times = transient_times[transient_times >= 0]
    
    # Cut down to every TSTEPth time point
    cut_clr_resp = [*map(lambda x: x[0::tstep,:],sub_clr_resp)]
    cut_ar_resp = [*map(lambda x: x[0::tstep,:],sub_ar_resp)]
    cut_times = sub_times[0::tstep]
    
    # Later, we'll need to scale GO/NO-GO activity by a weighting function. Let's define those weights here
    g_fun = lambda x, s, m: (.5*np.pi*(s**2))*np.exp((-.5)*((x-m)**2)/(s**2))/(.5*np.pi*(s**2))
    lar_scale_ng = g_fun(LAR_VALS, p['ng_width'], 0)
    
    w_fun = lambda x, b, k: 1 - np.exp(-1*((b*x)**k))
    lar_scale_go = w_fun(LAR_VALS,p['go_center'],p['go_shape'])
    lar_scale_go[LAR_VALS < 0] = 0
    
    # Initialize derived units
    v_colors = [np.empty([len(cut_times),len(CLR_VALS)])*np.nan for i in range(SET_SIZE)]
    sing_unit = np.empty([len(cut_times),SET_SIZE])*np.nan
    mov_unit = np.empty([len(cut_times),SET_SIZE])*np.nan
    go_unit = np.empty([len(cut_times),SET_SIZE])*np.nan
    ng_unit = np.empty([len(cut_times),SET_SIZE])*np.nan
    
    # Start them with randn
    for il in range(SET_SIZE):
        v_colors[il][0,:] = np.random.randn(1,len(CLR_VALS))*NOISE_SD + BL_FIRING
    sing_unit[0,:] = np.random.randn(1,SET_SIZE)*SING_NOISE_SD + BL_FIRING
    mov_unit[0,:] = np.random.randn(1,SET_SIZE)*SING_NOISE_SD + BL_FIRING
    go_unit[0,:] = np.random.randn(1,SET_SIZE)*SING_NOISE_SD + BL_FIRING
    ng_unit[0,:] = np.random.randn(1,SET_SIZE)*SING_NOISE_SD + BL_FIRING
    
    
    # Get ready for the loop
    stop_set = False
    stop_t = np.inf
    rt = np.nan
    
    # Start time step loop
    for it in range(1,len(cut_times)):
        
        if any(mov_unit[it-1,:] > MOV_THRESH) and not stop_set:
            stop_set = True
            stop_t = it + 2
            rt = cut_times[it-1]
        
        if it > stop_t:
            break
        
        # Next inner loop is looping over stimulus locations
        for il in range(SET_SIZE):
            
            # Finally, loop over the different color units
            for ic in range(cut_clr_resp[il].shape[1]):
                
                # If the other stimuli have the same color, this one can't be the singleton.
                # Calculate same_color_vals derived from other stimuli of the same color
                same_color_vals = sum([v_colors[i][it-1,ic] for i in range(SET_SIZE)]) - v_colors[il][it-1,ic]
                
                # If other stimuli have different colors, it's more likely that this one is the singleton
                # Calculate other_color_vals to sum up other_color_drive
                other_color_vals = sum([sum(v_colors[i][it-1,:])-v_colors[i][it-1,ic] for i in range(SET_SIZE) if i is not il]) 
    
                # Now lateral inhibition. If the spatially specific unit here says it's color ic,
                # then it can't be color not ic, right?
                lateral_inhib = sum(v_colors[il][it-1,:]) - v_colors[il][it-1,ic]
                
                # Now, the differential equation for the "ic as singleton" units
                dv_tmp = (cut_clr_resp[il][it,ic] - cut_clr_resp[il][it-1,ic]) + \
                    tstep*( \
                        other_color_vals*p['other_drive'] \
                            - same_color_vals*p['same_inhib'] \
                                -lateral_inhib*p['lat_colors'] \
                                    -v_colors[il][it-1,ic]*p['k'] \
                        ) + \
                        np.sqrt(tstep)*np.random.randn(1,1)*NOISE_SD
                v_colors[il][it,ic] = max((0,v_colors[il][it-1,ic]+dv_tmp))
            # END IC LOOP
            
            # Now that we have v_colors, all of these sum to drive the singleton units
            sing_drive = sum(v_colors[il][it,:])
            
            # But the singleton units should be inhibited by other singleton untis
            sing_inhib = sum(sing_unit[it-1,:]) - sing_unit[it-1,il]
            
            # Now write the differential equation
            ds_tmp = tstep * ( \
                sing_drive*p['sing_drive'] \
                - sing_inhib*p['sing_lateral'] \
                    + max((0, go_unit[it-1,il] - p['v_go_gate']))*p['go_vis_facil'] \
                        - max((0, ng_unit[it-1,il] - p['v_go_gate']))*p['ng_vis_inhib'] \
                            - sing_unit[it-1,il]*p['singk'] \
                ) + \
                np.sqrt(tstep)*np.random.randn(1,1)*SING_NOISE_SD
            sing_unit[it,il] = max((0, sing_unit[it-1,il] + ds_tmp))
            
            # Now, let's get the GO and NO-GO drive by applying the transfer functions to cut_ar_resp
            ng_drive = sum(np.multiply(cut_ar_resp[il][it-1,:],lar_scale_ng))
            go_drive = sum(np.multiply(cut_ar_resp[il][it-1,:],lar_scale_go))
            
            # Use this drive on the NO-GO units
            dn_tmp = tstep * ( \
                              ng_drive * p['stim_ng'] \
                                  - go_unit[it-1,il]*p['b_go'] \
                                      - ng_unit[it-1,il]*p['gngk'] \
                            ) + \
                np.sqrt(tstep)*np.random.randn(1,1)*SING_NOISE_SD
            ng_unit[it,il] = max((0, ng_unit[it-1,il]+dn_tmp))
            
            # And on the GO units
            dg_tmp = tstep * (
                go_drive * p['stim_go'] \
                    - ng_unit[it-1,il]*p['b_stop'] \
                        - go_unit[it-1,il]*p['gngk'] \
                ) + \
                np.sqrt(tstep)*np.random.randn(1,1)*SING_NOISE_SD
            go_unit[it,il] = max((0, go_unit[it-1,il]+dg_tmp))
            
            # Finally, we're ready to drive the mov units
            # Calculate feedforward and lateral inhibition
            ff_inhib = sum(np.multiply([p['ff_inhib'][i] for i in list(range(il,SET_SIZE))+list(range(il))],sing_unit[it-1,:]))
            lat_inhib = sum(np.multiply([p['lat_inhib'][i] for i in list(range(il,SET_SIZE))+list(range(il))],mov_unit[it-1,:]))
            
            # Set up the equation
            dm_tmp = tstep *( \
                max((0,(sing_unit[it-1,il] - ff_inhib - p['gate'])))*p['vm_drive'] \
                    + max((0, go_unit[it-1,il]-p['m_go_gate']))*p['go_mdrive'] \
                        - max((0, ng_unit[it-1,il]-p['m_ng_gate']))*p['ng_mdrive'] \
                            - lat_inhib \
                                - mov_unit[it-1,il]*p['k'] \
                ) + \
                np.sqrt(tstep)*np.random.randn(1,1)*SING_NOISE_SD
            mov_unit[it,il] = max((0, mov_unit[it-1,il]+dm_tmp))
        # END IL LOOP
    # END IT LOOP
    
    unit_activity = {'sing': sing_unit, 'mov': mov_unit, 'go': go_unit, 'ng': ng_unit}
    
    return unit_activity, rt, r_state_in
    

def GetTransient(gain=1, std=0, duration=800, bl_lead=-300, vis_latency=40):
    
    # Do imports
    import numpy as np
    
    # Set default variables
    TRANS_PEAK = 100
    SUST_RESP = 80
    TG = 10
    TD = 40
    BASELINE = 10
    OFF_TIME = 500

    # Set flag for doing Poisson-type noise. Not yet supported
    POISS_NOISE = False
    
    
    # Now generate the base SDF
    in_resp_sdf = (1-np.exp(-1*(np.arange(0,duration))/TG))*np.exp(-1*(np.arange(0,duration))/TD)
    # Scale it to max out at 1
    in_resp_sdf = in_resp_sdf/max(in_resp_sdf)
    
    # Generate the rising and falling pieces
    piece_rise = in_resp_sdf*(TRANS_PEAK - BASELINE)
    piece_fall = in_resp_sdf*((TRANS_PEAK - BASELINE) - SUST_RESP) + (SUST_RESP-BASELINE)
    model_resp = np.max(np.vstack((piece_rise,piece_fall)),axis=0)
    
    # Combine baseline for VIS_LATENCY samples at the front
    sdf = np.concatenate((np.zeros([vis_latency,1]),model_resp.reshape(-1,1)),axis=0)
    sdf = sdf[0:duration]
    t = np.arange(0,duration)
    sdf[t > OFF_TIME] = 0
    
    # The above works as though the beginning is t=0. We want to cut or expand as needed to account for the baseline period
    if bl_lead > 0:
        sdf = sdf[t >= bl_lead]
        t = t[t >= bl_lead]
    else:
        n_add = 0-bl_lead
        sdf = np.concatenate((np.zeros([n_add,1]),sdf),axis=0)
        t = np.concatenate((np.arange(bl_lead,0),t))
    
    # Now we need to add noise to the trace
    if POISS_NOISE:
        # Not yet supported, leaving comment for placeholder
        tmp = 0
    else:
        sdf = (sdf + np.random.randn(sdf.shape[0],sdf.shape[1])*std)*gain
    
    # Finally, reintroduce the baseline
    sdf = sdf + BASELINE
    sdf[sdf < 0] = abs(sdf[sdf < 0])
     
    return sdf, t


def DecodeTrialType(type_str):
    # type_str should be 'hh','hl',lh','ll','h0', or 'l0'. The first position is identifiability, second is discriminability
    # Set up a dict that can be indexed easily
    hl_dict = {'h': True, 'l': False, '0': False}
    gng_dict = {'h': True, 'l': True, '0': False}
    
    is_hi = hl_dict[type_str[0].lower()]
    is_hd = hl_dict[type_str[1].lower()]
    is_go = gng_dict[type_str[1].lower()]
    
    return is_hi, is_hd, is_go


def GetGainGauss(g_std, offset):
    # Do imports
    import numpy as np
    
    # We're getting a Gaussian centered at 180, with an SD of g_std, evaluated at 1:360 degrees
    
    # Make the vector for evaluation
    e_points = np.arange(0,360)
    
    # Make a lambda function to evaluate the gaussian
    g_fun = lambda x, s: (.5*np.pi*(s**2))*np.exp((-.5)*((x-180)**2)/(s**2))/(.5*np.pi*(s**2))
    g_vals = g_fun(e_points, g_std)
    
    # Make sure the offset > 0...
    while offset < 0:
        offset = offset + 360
    # and offset <= 360
    while offset > 360:
        offset = offset - 360
    
    # Wrap the function by finding where e_points > offset first
    over_offset = [i for i in range(len(e_points)) if e_points[i] >= offset]
    cut_ind = over_offset[0]
    
    # Now, slice the values
    gain_gauss = np.concatenate((g_vals[cut_ind:].reshape(-1,1),g_vals[0:cut_ind].reshape(-1,1)),axis=0)
    
    return gain_gauss


def SetupClrTransients(stim_colors, tr_duration=600, clr_step = 10, clr_std = 5, bl_start=-300):
    # Do imports
    import numpy as np
    
    # Make a vector of center points that can be indexed into
    CENTER_VALS = [0, 180, 70, 120]
    CLR_WIDTH = 45
    BASE_GAIN = .2
    
    # Make a vector of color values
    clr_vals = np.arange(0,360,clr_step)

    # Set up output
    clr_transients = []
    
    # Loop through stim_colors
    for i in range(len(stim_colors)):
        # Get the Gain Gaussian
        gain_gauss = GetGainGauss(CLR_WIDTH, CENTER_VALS[stim_colors[i]])
        # Now scale it
        gain_gauss = (gain_gauss*(1-BASE_GAIN)) + BASE_GAIN
     
        base_transient = np.empty((len(range(bl_start,tr_duration)),len(clr_vals)))
        for ic, cv in enumerate(clr_vals):
            [tmp_sdf, tmp_t] = GetTransient(gain_gauss[cv],std=clr_std, duration=tr_duration, bl_lead=bl_start)
            base_transient[:,ic] = tmp_sdf.ravel()
     
        clr_transients.append(base_transient)
        
    return clr_transients


def SetupShapeTransients(stim_elongs, tr_duration=600, ar_step=.05, ar_std=5, bl_start=-300):
    # Do imports
    import numpy as np
    
    # Make a vector of center points that can be indexed into
    CENTER_VALS = [1.0, 1.22, 2.0]
    AR_WIDTH = 0.2
    BASE_GAIN = .2
    
    # Make a vector of color values
    ar_vals = np.arange(-1,1+ar_step,ar_step)

    # Prep Gaussian function    
    g_fun = lambda x, s, m: (.5*np.pi*(s**2))*np.exp((-.5)*((x-m)**2)/(s**2))/(.5*np.pi*(s**2))
    
    # Set up output
    ar_transients = []
    
    # Loop through stim_elongs
    for i in range(len(stim_elongs)):
        base_transient = np.empty((len(range(bl_start,tr_duration)),len(ar_vals)))
        for ic, sv in enumerate(ar_vals):
            [tmp_sdf, tmp_t] = GetTransient(g_fun(sv,AR_WIDTH, np.log(CENTER_VALS[stim_elongs[i]]))*(1-BASE_GAIN)+BASE_GAIN, std=ar_std, duration=tr_duration, bl_lead=bl_start)
            base_transient[:,ic] = tmp_sdf.ravel()
     
        ar_transients.append(base_transient)
        
    return ar_transients


def GetCDF(in_distro, min_val = None, max_val = None, step = 1):
    
    import math
    
    # Get min/max for range of data (unless otherwise specified
    min_val = math.floor(min(in_distro)) if min_val is None else min_val
    max_val = math.ceil(max(in_distro)) if max_val is None else max_val
    
    bins = [i/10 for i in range(min_val*10,max_val*10,step*10)]
    cdf = [sum(in_distro <= bins[i])/len(in_distro) for i in range(len(bins))]
    
    return cdf, bins


def GetSparkContext(core_prop = 1):
    import pyspark as ps
    import os
    
    # Get number of cores
    try:
        n_cores = len(os.sched_getaffinity(0))
    except:
        import psutil
        n_cores = psutil.cpu_count()
        
    n_str = 'local[' + str(int(n_cores*core_prop)) + ']'
    
    spark = ps.sql.SparkSession.builder \
    .master(n_str) \
        .appName('spark-ml') \
            .getOrCreate()
    sc = spark.sparkContext

    return sc


if __name__ == '__main__':
    main()