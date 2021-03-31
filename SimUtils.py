# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 07:33:56 2021

@author: klowe
"""

def GetTransient(gain=1, std=0, duration=800, bl_lead=-300, VIS_LATENCY=40):
    
    # Do imports
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Set default variables
    TRANS_PEAK = 100
    SUST_RESP = 80
    VIS_LATENCY = 40
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
    sdf = np.concatenate((np.zeros([VIS_LATENCY,1]),model_resp.reshape(-1,1)),axis=0)
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
    hl_dict = {'h': True, 'l': False}
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


def SetupClrTransients(stim_colors, TR_DURATION=600, CLR_STEP = 10, CLR_STD = 5, BL_START=-300):
    # Do imports
    import numpy as np
    
    # Make a vector of center points that can be indexed into
    CENTER_VALS = [0, 180, 70, 120]
    CLR_WIDTH = 45
    BASE_GAIN = .2
    
    # Make a vector of color values
    clr_vals = np.arange(0,360,CLR_STEP)

    # Set up output
    clr_transients = []
    
    # Loop through stim_colors
    for i in range(len(stim_colors)):
        # Get the Gain Gaussian
        gain_gauss = GetGainGauss(CLR_WIDTH, CENTER_VALS[stim_colors[i]])
        # Now scale it
        gain_gauss = (gain_gauss*(1-BASE_GAIN)) + BASE_GAIN
     
        base_transient = np.empty((len(range(BL_START,TR_DURATION)),len(clr_vals)))
        for ic, cv in enumerate(clr_vals):
            [tmp_sdf, tmp_t] = GetTransient(gain_gauss[cv],std=CLR_STD, duration=TR_DURATION, bl_lead = BL_START)
            base_transient[:,ic] = tmp_sdf.ravel()
     
        clr_transients.append(base_transient)
        
    return clr_transients


def SetupShapeTransients(stim_elongs, TR_DURATION=600, AR_STEP = .05, AR_STD = 5, BL_START=-300):
    # Do imports
    import numpy as np
    
    # Make a vector of center points that can be indexed into
    CENTER_VALS = [1.0, 1.22, 2.0]
    AR_WIDTH = 0.2
    BASE_GAIN = .2
    
    # Make a vector of color values
    ar_vals = np.arange(-1,1+AR_STEP,AR_STEP)

    # Prep Gaussian function    
    g_fun = lambda x, s, m: (.5*np.pi*(s**2))*np.exp((-.5)*((x-m)**2)/(s**2))/(.5*np.pi*(s**2))
    
    # Set up output
    ar_transients = []
    
    # Loop through stim_elongs
    for i in range(len(stim_elongs)):
        base_transient = np.empty((len(range(BL_START,TR_DURATION)),len(ar_vals)))
        for ic, sv in enumerate(ar_vals):
            [tmp_sdf, tmp_t] = GetTransient(g_fun(sv,AR_WIDTH, np.log(CENTER_VALS[stim_elongs[i]]))*(1-BASE_GAIN)+BASE_GAIN, std = AR_STD, duration=TR_DURATION, bl_lead=BL_START)
            base_transient[:,ic] = tmp_sdf.ravel()
     
        ar_transients.append(base_transient)
        
    return ar_transients


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


def GetCDF(in_distro, min_val = None, max_val = None, step = 1):
    
    import math
    
    # Get min/max for range of data (unless otherwise specified
    min_val = math.floor(min(in_distro)) if min_val is None else min_val
    max_val = math.ceil(max(in_distro)) if max_val is None else max_val
    
    bins = [i/10 for i in range(min_val*10,max_val*10,step*10)]
    cdf = [sum(in_distro <= bins[i])/len(in_distro) for i in range(len(bins))]
    
    return cdf, bins

    
def SimTrialLoop(trial_type, TSTEP=5):

    # Do imports
    import numpy as np
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
    clr_transients = SetupClrTransients(stim_colors.astype(int), TR_DURATION=TR_DUR, CLR_STEP=CLR_STEP, CLR_STD = CLR_STD, BL_START = BL_START)
    ar_transients = SetupShapeTransients(stim_elongs.astype(int), TR_DURATION=TR_DUR, AR_STEP=AR_STEP, AR_STD = AR_STD, BL_START = BL_START)
    transient_times = np.arange(BL_START,TR_DUR)
    
    # Cut down to just t > 0
    sub_clr_resp = [*map(lambda x: x[transient_times >= 0,:], clr_transients)]
    sub_ar_resp = [*map(lambda x: x[transient_times >= 0,:], ar_transients)]
    sub_times = transient_times[transient_times >= 0]
    
    # Cut down to every TSTEPth time point
    cut_clr_resp = [*map(lambda x: x[0::TSTEP,:],sub_clr_resp)]
    cut_ar_resp = [*map(lambda x: x[0::TSTEP,:],sub_ar_resp)]
    cut_times = sub_times[0::TSTEP]
    
    # Later, we'll need to scale GO/NO-GO activity by a weighting function. Let's define those weights here
    g_fun = lambda x, s, m: (.5*np.pi*(s**2))*np.exp((-.5)*((x-m)**2)/(s**2))/(.5*np.pi*(s**2))
    lar_scale_ng = g_fun(LAR_VALS, p['ng_width'], 0)
    
    w_fun = lambda x, b, k: 1 - np.exp(-1*((b*x)**k))
    lar_scale_go = w_fun(LAR_VALS,p['go_center'],p['go_shape'])
    lar_scale_go[LAR_VALS < 0] = 0
    
    # Initialize derived units
    v_colors = [np.empty([len(cut_times),len(CLR_VALS)]) for i in range(SET_SIZE)]
    sing_unit = np.empty([len(cut_times),SET_SIZE])
    mov_unit = np.empty([len(cut_times),SET_SIZE])
    go_unit = np.empty([len(cut_times),SET_SIZE])
    ng_unit = np.empty([len(cut_times),SET_SIZE])
    
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
                    TSTEP*( \
                        other_color_vals*p['other_drive'] \
                            - same_color_vals*p['same_inhib'] \
                                -lateral_inhib*p['lat_colors'] \
                                    -v_colors[il][it-1,ic]*p['k'] \
                        ) + \
                        np.sqrt(TSTEP)*np.random.randn(1,1)*NOISE_SD
                v_colors[il][it,ic] = max((0,v_colors[il][it-1,ic]+dv_tmp))
            # END IC LOOP
            
            # Now that we have v_colors, all of these sum to drive the singleton units
            sing_drive = sum(v_colors[il][it,:])
            
            # But the singleton units should be inhibited by other singleton untis
            sing_inhib = sum(sing_unit[it-1,:]) - sing_unit[it-1,il]
            
            # Now write the differential equation
            ds_tmp = TSTEP * ( \
                sing_drive*p['sing_drive'] \
                - sing_inhib*p['sing_lateral'] \
                    + max((0, go_unit[it-1,il] - p['v_go_gate']))*p['go_vis_facil'] \
                        - max((0, ng_unit[it-1,il] - p['v_go_gate']))*p['ng_vis_inhib'] \
                            - sing_unit[it-1,il]*p['singk'] \
                ) + \
                np.sqrt(TSTEP)*np.random.randn(1,1)*SING_NOISE_SD
            sing_unit[it,il] = max((0, sing_unit[it-1,il] + ds_tmp))
            
            # Now, let's get the GO and NO-GO drive by applying the transfer functions to cut_ar_resp
            ng_drive = sum(np.multiply(cut_ar_resp[il][it-1,:],lar_scale_ng))
            go_drive = sum(np.multiply(cut_ar_resp[il][it-1,:],lar_scale_go))
            
            # Use this drive on the NO-GO units
            dn_tmp = TSTEP * ( \
                              ng_drive * p['stim_ng'] \
                                  - go_unit[it-1,il]*p['b_go'] \
                                      - ng_unit[it-1,il]*p['gngk'] \
                            ) + \
                np.sqrt(TSTEP)*np.random.randn(1,1)*SING_NOISE_SD
            ng_unit[it,il] = max((0, ng_unit[it-1,il]+dn_tmp))
            
            # And on the GO units
            dg_tmp = TSTEP * (
                go_drive * p['stim_go'] \
                    - ng_unit[it-1,il]*p['b_stop'] \
                        - go_unit[it-1,il]*p['gngk'] \
                ) + \
                np.sqrt(TSTEP)*np.random.randn(1,1)*SING_NOISE_SD
            go_unit[it,il] = max((0, go_unit[it-1,il]+dg_tmp))
            
            # Finally, we're ready to drive the mov units
            # Calculate feedforward and lateral inhibition
            ff_inhib = sum(np.multiply([p['ff_inhib'][i] for i in list(range(il,SET_SIZE))+list(range(il))],sing_unit[it-1,:]))
            lat_inhib = sum(np.multiply([p['lat_inhib'][i] for i in list(range(il,SET_SIZE))+list(range(il))],mov_unit[it-1,:]))
            
            # Set up the equation
            dm_tmp = TSTEP *( \
                max((0,(sing_unit[it-1,il] - ff_inhib - p['gate'])))*p['vm_drive'] \
                    + max((0, go_unit[it-1,il]-p['m_go_gate']))*p['go_mdrive'] \
                        - max((0, ng_unit[it-1,il]-p['m_ng_gate']))*p['ng_mdrive'] \
                            - lat_inhib \
                                - mov_unit[it-1,il]*p['k'] \
                ) + \
                np.sqrt(TSTEP)*np.random.randn(1,1)*SING_NOISE_SD
            mov_unit[it,il] = max((0, mov_unit[it-1,il]+dm_tmp))
        # END IL LOOP
    # END IT LOOP
    
    unit_activity = {'sing': sing_unit, 'mov': mov_unit, 'go': go_unit, 'ng': ng_unit}
    
    return unit_activity, rt
    