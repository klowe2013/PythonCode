# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:52:09 2021

@author: klowe
"""

def main():
    # Basic imports
    import pickle
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    
    # Now prep the Keras stuff
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.callbacks import EarlyStopping
    from keras.models import load_model

    types = ['hh','hl','lh','ll','h0','l0']
        
    saved_obj = pickle.load(open("./testSave.obj",'rb'))
    
    unit_activities = saved_obj[0]
    rts = saved_obj[1]
    # types = saved_obj[2]
    
    #unit_activities = saved_obj['Units']
    #rts = saved_obj['RTs']
    #types = saved_obj['Types']
    
    # Set hyperparameters
    WINDOW = 75
    WHICH_COND_TRAIN = ['hh', 'hl']
    TRAIN_PROP = .7
    N_TRS = len(unit_activities[0])
    DROPOUT_RATE = .2
    
    # For each condition, pull singleton and mov unit activities
    all_activities = [[] for i in range(len(types))]
    out_rf_s = [[] for i in range(len(types))]
    in_rf_m = [[] for i in range(len(types))]
    out_rf_m = [[] for i in range(len(types))]
    all_preds = [[] for i in range(len(types))]
    all_movs = [[] for i in range(len(types))]
    for i in range(len(unit_activities)):
        sing_activities = [unit_activities[i][ii]['sing'] for ii in range(len(unit_activities[i]))]
        mov_activities = [unit_activities[i][ii]['mov'] for ii in range(len(unit_activities[i]))]
        go_activities = [unit_activities[i][ii]['go'] for ii in range(len(unit_activities[i]))]
        ng_activities = [unit_activities[i][ii]['ng'] for ii in range(len(unit_activities[i]))]
        
        all_activities[i] = [[] for it in range(len(sing_activities))]
        all_activities[i][0] = np.hstack((sing_activities[0],go_activities[0],ng_activities[0],mov_activities[0]))
        for it in range(1,len(sing_activities)):
            nan_locs = np.where(np.isnan(sing_activities[it][:,0]))[0]
            if len(nan_locs) > 0:
                first_nan = nan_locs[0]
            else:
                first_nan = sing_activities[it][:,0].shape[0]
            all_activities[i][it] = np.hstack((sing_activities[it],go_activities[it],ng_activities[it],mov_activities[it]))
            
    cond_train_ind = [i for i in range(len(types)) if types[i] in WHICH_COND_TRAIN]
    
    rand_order = random.sample(list(range(N_TRS)), N_TRS)
    train_trials = rand_order[:int(N_TRS*TRAIN_PROP)]
    test_trials = rand_order[int(N_TRS*TRAIN_PROP):]
    all_preds = []
    for i in range(len(cond_train_ind)):
        for it in range(len(train_trials)):
            preds_tmp, movs_tmp = TrialToTraining(all_activities[cond_train_ind[i]][train_trials[it]], WINDOW)
            try:
                all_preds = np.vstack((all_preds,preds_tmp))
                all_movs = np.vstack((all_preds,movs_tmp))
            except:
                all_preds = preds_tmp
                all_movs = movs_tmp
    all_preds = np.array(all_preds)
    all_movs = np.array(all_movs)
    
    # Initialize the model
    m = Sequential()
    m.add(LSTM(units=50, return_sequences=True, input_shape=(all_preds.shape[1],all_preds.shape[2])))
    m.add(Dropout(DROPOUT_RATE))
    m.add(LSTM(units=50))
    m.add(Dropout(DROPOUT_RATE))
    m.add(Dense(units=all_movs.shape[1]))
    m.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fit the model
    history = m.fit(all_preds, all_movs, epochs = 10, batch_size = 250, verbose=1)


def TrialToTraining(trial_activities,window):
    # Store window number of points as a sequence
    pred_vals = []#trial_activities[0:window,:-8]
    mov_vals = []#trial_activities[window,-8:]
    for i in range(window,trial_activities.shape[0]):
        pred_vals.append(trial_activities[i-window:i,:-8])
        mov_vals.append(trial_activities[i,-8:])

    return pred_vals, mov_vals
    

if __name__ == '__main__':
    main()