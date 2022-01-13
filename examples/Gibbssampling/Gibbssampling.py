#!/usr/bin/env python


import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from pathlib import Path, PurePath, PureWindowsPath
from tqdm.notebook import tqdm, trange
import os, sys
sys.path.append(r"/")
import util, flim


# In[3]:


irf = util.SDT(Path(r"IRF-M2.sdt"), channel=[0])
irf.decay_curve(plot=True, channel=[0], trunc=True, bgsub=True, peak_start=2.75, peak_end=3.75)


# In[12]:


dg = flim.decay_group(np.zeros(256),irf)
dg.fit_start_bin, dg.fit_end_bin, dg.nphot_in_range = 12, 240, 1000000


truth = np.zeros(256)
truth[(dg.fit_start_bin):(dg.fit_end_bin+1)] = dg.model_biexp(20.5,0.995,3.5,0.4,0.25)*1e3
print(np.sum(truth))
truth_norm = truth / np.sum(truth)

# plt.figure(figsize=(27,3))
# plt.plot(truth, "o", markersize=3)
# plt.axvline(x=12)
# plt.axvline(x=240)


dg = flim.decay_group(truth,irf)
fitp = dg.fit(save_leastsq_params_array=True)
# dg.plot()
print(fitp)

param_samples = dg.gibbs_sample([True, False, False, False, False], showprogress=True)


dg.params["A"]["value"] += 0.01
dg.params["tau1"]["value"] += 0.3
dg.params["tau2"]["value"] -= 0.3
dg.params["f"]["value"] += 0.2


# In[48]:


ndrawslist = [10000] # np.logspace(3,6,20).astype(int)
nreps = 1
for i_ndraws in range(len(ndrawslist)):
    ndraws = ndrawslist[i_ndraws]
    for i in range(nreps):
        cts, bes = np.histogram(np.random.choice(256, size=(ndraws), p=truth_norm), bins=-0.5+np.arange(257))
        dg.load_data(cts)
        param_samples = dg.gibbs_sample([True, False, False, False, False], nburn=100, nsample=500, showprogress=True)

        # dg.params["A"]["value"] -= 0.01
        # dg.params["tau1"]["value"] += 0.2
        # dg.params["tau2"]["value"] -= 0.2
        # dg.params["f"]["value"] += 0.2
        #
        # param_samples2 = dg.gibbs_sample([True, False, False, False, False], nburn=0, nsample=500)
# #         np.savetxt(f"draws={ndraws:d}_rep={i:d}.txt", param_samples)d.

