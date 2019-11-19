# Reinforcement Learning models with precision-weighting of prediction errors

Matlab analyses code used for Rescorla Wagner (RW) and Pearce-Hall (PH) Reinforcement Learning (RL) models in "Precision-weighting of cortical unsigned prediction error signals benefits learning, is mediated by dopamine, and is impaired in psychosis. J Haarsma, PC Fletcher, JD Griffin, HJ Taverne, H Ziauddeen, TJ Spencer, C Miller, T Katthagen, I Goodyer, KMJ Diederen*, GK Murray*"; *equal contribution: https://www.biorxiv.org/content/10.1101/558478v1.abstract

This code contains the following scripts:
1. pesd_fits_no_scaling.m: Standard RW and PH RL models
2. pesd_fits_scaling.m: RW and PH RL models with a fixed scaling term for the Prediction Error (PE) based on the precision of task conditions
3. pesd_fits_est_scaling.m: PH RL model with a free parameter to scale the PE based on the precision of task conditions. Please note the RW is the same as in 1. 
4. pesd_fits_est_scaling_2param.m: PH RL model with two free parameters used to seperately scale signed and unsigned PEs based on the precision of task conditions. The RW is the same as in 1. 

We would like to acknowledge Dr Martin Vestergaard for the initail versions of the models as published in https://www.sciencedirect.com/science/article/pii/S0896627316300782 and https://www.jneurosci.org/content/37/7/1708?utm_source=TrendMD&utm_medium=cpc&utm_campaign=JNeurosci_TrendMD_0
 
Please feel welcome to use and reproduce this code as you see fit. We apologise for the limited commenting. Please feel free to ask any questions and add corrections and or comments.
