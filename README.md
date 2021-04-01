# PythonCode
Repository for python code for various sub-projects

Projects:
1) gngSearchSimulation - Simulates biologically plausible units corresponding to FEF vis/mov neurons during the GO/NO-GO search task
	This is a go/no-go extension of the Gated Accumulator Model (Purcell et al., 2012) along with a pop-out mechanism
	All functions are included with a loop over trial types/trial numbers being in main(). Plots CDFs, SIC, and average vis/mov unit activity
	
2) SDF_Maker - This was a test project to write an SDF convolver, ported from MATLAB. 
	This needs the klGenFuns module from the base directory. 
	The .mat file inside contains the behavior and spike time variables that get used to make the SDF
	
3) SST_Stats - This file loads an excel file with SST/CDT times (not included due to lab policy, for the moment), and runs stats on the selection time measures

In the base directory, some utility functions are included in klGenFuns.python
