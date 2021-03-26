# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:06:06 2021

@author: klowe
"""

# Standard imports
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Bring in random forsest stuff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# Set global parameters
# Unit inclusion
VIS_MIN = 1
VIS_MAX = 2.5
MOV_MIN = 2.5
MOV_MAX = 3
MIN_QUAL = 3

# Model fitting
DO_SCALE = True
TEST_PROP = 0.2        # What is the proportion of values used in the test set?
N_ITER = 500               # How many random iterations of train/test splits do we want?
DO_RANDOM_ITER = True
N_ESTIMATORS = 20
MIN_SAMPS = .01
WHICH_REGRESS = 'log' # 'log' or 'forest', defaults to 'log'

# Define functions used in script below
def do_ident_discrim_anova(in_mat):
    import statsmodels.formula.api as smf
    
    sst_mat = in_mat.iloc[:,:4]
    sst_mat = sst_mat[~np.any(pd.isnull(sst_mat),axis=1)]
    
    clr_mat = np.concatenate([np.ones([sst_mat.shape[0],2]),np.ones([sst_mat.shape[0],2])*2],axis=1)
    ar_mat = np.concatenate([np.ones([sst_mat.shape[0],1]),np.ones([sst_mat.shape[0],1])*2],axis=1)
    ar_mat = np.concatenate([ar_mat,ar_mat],axis=1)
    in_vals = sst_mat.values
    unit_mat = np.transpose(np.tile(np.arange(0,sst_mat.shape[0]),[4,1]))

    n_elements = 1;
    for dim in np.shape(in_vals):
        n_elements *= dim

    data_mat = np.concatenate([np.reshape(in_vals,[n_elements,1]),np.reshape(clr_mat,[n_elements,1]),np.reshape(ar_mat,[n_elements,1]),np.reshape(unit_mat,[n_elements,1])],axis=1)
    sst_stats_frame = pd.DataFrame({'SST': data_mat[:,0], 'Ident': data_mat[:,1], 'Discrim': data_mat[:,2], 'Unit': data_mat[:,3]})
    sst_stats_frame['SST'] = pd.to_numeric(sst_stats_frame['SST'])
    
    model = smf.mixedlm('SST ~ C(Ident)*C(Discrim)', data=sst_stats_frame, groups=sst_stats_frame['Unit'])
    model_fit = model.fit()
    p_vals = model_fit.pvalues[1:]
    
    return model_fit, p_vals


def make_id_errorbars(in_frame, p=[1,1,1], title_label='Unk'):
    
    sst_mat = in_frame.iloc[:,:4]
    
    
    mu_sst = np.mean(sst_mat)
    sd_sst = np.std(sst_mat)/np.sqrt(in_frame.shape[0])
    
    plt.figure()
    plt.errorbar([1,2],mu_sst[0:2],yerr=sd_sst[0:2],color=[0, 0, 0])
    plt.errorbar([1,2],mu_sst[2:4],yerr=sd_sst[2:4],color=[.8, .2, .2])
    
    if in_frame.shape[1] > 4:
        rt_mat = in_frame.iloc[:,4:]
        mu_rt = np.mean(rt_mat)
        sd_rt = np.std(rt_mat)/np.sqrt(in_frame.shape[0])
        plt.errorbar([1,2],mu_rt[0:2],yerr=sd_rt[0:2],color=[0, 0, 0])
        plt.errorbar([1,2],mu_rt[2:4],yerr=sd_rt[2:4],color=[.8, .2, .2])
    
    title_str = '%s: p(I)=%.3f, p(D)=%.3f, p(X)=%.3f' % (title_label,p[0],p[1],p[2])
    
    plt.xticks([1,2],labels=['H_D','L_D'])
    plt.xlabel('Discriminability')
    plt.ylabel('Time From Array (ms)');
    plt.title(title_str)
    plt.show()
    
    
# Make sure we're in the right directory
os.chdir('C:\\Users\\klowe\\Dropbox\\Schall-Lab\\Papers\\NeuralSFT\\DataCuration\\')

# Read the text file
df = pd.read_excel('./Neural SFT -- SelectionTimes_AllUnits 22-March-2021.xlsx')

# Cut padding at the top of the file
df_sub = df.iloc[5:,:]
col_names = df.iloc[4,:]
df_sub.rename(columns=col_names,inplace=True)

df_sub = df_sub.loc[(df_sub.iloc[:,7] <= MIN_QUAL),:]

is_mov = (df_sub.iloc[:,6] <= MOV_MAX) & (df_sub.iloc[:,6] >= MOV_MIN)
is_vis = df_sub.iloc[:,6] <= VIS_MAX
is_co = df_sub.loc[:,'IsCoactive'] == 1

sst_cols = ['SST_HH','SST_HL','SST_LH','SST_LL','RT_HH','RT_HL','RT_LH','RT_LL']
cdt_cols = ['CDT_HH','CDT_HL','CDT_LH','CDT_LL','RT_HH','RT_HL','RT_LH','RT_LL']

my_stats = [(is_vis & ~is_co), (is_vis & is_co), (is_mov & ~is_co), (is_mov & is_co)]
cond_labs = ['v-NonCo','v-Co','m-NonCo','m-Co']

sst_aov =  []
cdt_aov = []
for i in range(len(my_stats)):
    this_sst_df = df_sub[my_stats[i]].loc[:,sst_cols]
    model, p_vals = do_ident_discrim_anova(this_sst_df)
    sst_aov.append(model)
    make_id_errorbars(this_sst_df, p=p_vals, title_label='SST'+cond_labs[i])
    
    this_cdt_df = df_sub[my_stats[i]].loc[:,cdt_cols]
    model, p_vals = do_ident_discrim_anova(this_cdt_df)
    cdt_aov.append(model)
    make_id_errorbars(this_cdt_df, p=p_vals, title_label='CDT'+cond_labs[i])
    
# Set up features for regression
my_measures = ['VL','SST','CDT','MovRamp','MovRampSST']
my_features = ['IsCoactive']
for i in range(len(my_measures)):
    # Add the HH, HL, LH, and LL conditions to my_features for column indexing
    my_features.append(my_measures[i]+'_HH')
    my_features.append(my_measures[i]+'_HL')
    my_features.append(my_measures[i]+'_LH')
    my_features.append(my_measures[i]+'_LH')
    my_features.append(my_measures[i]+'_MIC')
    
    # Calculate the MIC
    these_cols = [ic for ic in range(len(col_names)) if type(col_names[ic]) is str and col_names[ic][0:len(my_measures[i])] == my_measures[i]]
    this_mic = (df_sub.iloc[:,these_cols[0]] - df_sub.iloc[:, these_cols[1]])- (df_sub.iloc[:,these_cols[2]] - df_sub.iloc[:,these_cols[3]])
    df_sub[(my_measures[i]+'_MIC')] = this_mic
    
df_features = df_sub.loc[:,my_features]
df_features['IsVis'] = is_vis
df_features['IsMov'] = is_mov
my_features.append('IsVis')
my_features.append('IsMov')

# Scale the responses using StandardScaler
if DO_SCALE:
    scaler = StandardScaler()
    all_vals = scaler.fit_transform(df_features)
else:
    all_vals = df_features.values()

# Now the first column of all_vals is the label, 2nd-end columns are predictors. Separate them
all_preds = all_vals[:,1:]
all_labels = all_vals[:,0]

# Assign all nans in all_preds to 0 (the mean of the samples, given that StandardScaler is a Z score)
all_preds[np.isnan(all_preds)] = 0
    
# Initialize output vectors for fit quality to be populated by N_ITER loop
train_r2 = []
train_mse = []
test_r2 = []
test_mse = []

# Now start the if statement logic for which regressor to use
# If 'forest' do RandomForest
if WHICH_REGRESS.lower() == 'forest':
    
    # Start an interation loop, doing N_ITER different train/test splits
    for ii in range(N_ITER):
        # Print output if we want progress reports every 20 iterations
        if ii % 20 == 0:
            print('Working on iteration %d' % ii)
        
        if DO_RANDOM_ITER:
            random_start_val = ii
        else:
            random_start_val = None
            
        # Train/test split
        train_preds, test_preds, train_labs, test_labs = train_test_split(all_preds, all_labels, test_size=TEST_PROP, random_state=random_start_val)
        
        # Initialize and train the model
        my_model = RandomForestRegressor(n_estimators = N_ESTIMATORS, random_state = None, min_samples_leaf = MIN_SAMPS)
        my_model.fit(train_preds, train_labs)
        
        # Evaluate fit quality by predicting the training set
        fit_labs_train = my_model.predict(train_preds)
        
        # Now calculate r2 and mse
        train_r2.append(r2_score(train_labs,fit_labs_train))
        train_mse.append(mean_squared_error(train_labs,fit_labs_train))
        
        # Now try fitting the test set
        fit_labs_test = my_model.predict(test_preds)
        test_r2.append(r2_score(test_labs, fit_labs_test))
        test_mse.append(mean_squared_error(test_labs, fit_labs_test))

# Else if 'log' or 'linear' set up log/linear (separated within the syntax below)
elif WHICH_REGRESS.lower() == 'log' | WHICH_REGRESS:
    # For this, use only MIC values because random forest won't subselect features
    mic_features = [(i-1) for i in range(len(my_features)) if ("MIC" in my_features[i]) | ("IsVis" in my_features[i]) | ('IsMov' in my_features[i])]
    mic_preds = all_preds[:,mic_features]
    
    for ii in range(N_ITER):
        # Print output if we want progress reports every 100 iterations
        if ii % 100 == 0:
            print('Working on iteration %d' % ii)
        
        if DO_RANDOM_ITER:
            random_start_val = ii
        else:
            random_start_val = None
         
        # Train/test split
        train_preds, test_preds, train_labs, test_labs = train_test_split(mic_preds, df_features['IsCoactive'].values, test_size=TEST_PROP, random_state=random_start_val)
        
        if WHICH_REGRESS.lower() == 'log':
            my_model = LinearRegression()
        elif WHICH_REGRESS.lower() == 'linear':
            my_model = LinearRegression()
        my_model.fit(train_preds,train_labs.astype('int'))
        
        # Evaluate fit quality by predicting the training set
        fit_labs_train = my_model.predict(train_preds)
        
        # Now calculate r2 and mse
        train_r2.append(r2_score(train_labs,fit_labs_train))
        train_mse.append(mean_squared_error(train_labs,fit_labs_train))
        
        # Now try fitting the test set
        fit_labs_test = my_model.predict(test_preds)
        test_r2.append(r2_score(test_labs, fit_labs_test))
        test_mse.append(mean_squared_error(test_labs, fit_labs_test))
    
    
'''
# Get column names including "Max-Min/AVG"
ratio_inds = [i for i in range(len(col_names)) if col_names[i] == 'Max-Min/AVG']
for i in range(len(ratio_inds)):
    plt.figure()
    sns.distplot(abs(df_sub.iloc[:,ratio_inds[i]]),bins=np.arange(-1,1.05,.05))
    this_value = col_names[ratio_inds[i]-1][col_names[ratio_inds[i]-1].index('_')+1:]
    plt.title(this_value)
    plt.xlim([-.5,1.1])

'''