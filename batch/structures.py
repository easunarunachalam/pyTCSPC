import util, illprof, flim
from segmentation import segment, hungarian

import pandas as pd
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt

class mcimg(object):

    def __init__(self, sdt_fns, idx_in_main_list=np.nan, load=True, **kwargs):
        self.ch_fns     = sdt_fns
        self.ch         = [None for i in self.ch_fns]
        self.ch_int_fns = [util.SDT(ch_fn, noload=True).filename_int_corr for ch_fn in self.ch_fns]
        self.ch_int     = [None for i in self.ch_fns]
        self.mask_final = [None for i in self.ch_fns]
        self.idx_main   = idx_in_main_list

        if load:
            # pass parameters for load function
            self.load_data(**kwargs)

    def load_data(self, reload=False, loadtaudim=True):

        for i in range(len(self.ch_fns)):
            if reload or (self.ch[i] is None):
                if loadtaudim:
                    self.ch[i] = util.SDT(self.ch_fns[i])
                    self.ch_int[i] = self.ch[i].image()[1]
                else:
                    self.ch_int[i] = iio.imread(str(self.ch_int_fns[i]))

    def merge_channels(self, plot=False, plotname="allchannels.pdf"):

        self.zero = np.zeros_like(self.ch_int[0])
        print("self.zero shape = ", self.zero.shape)
        self.ch0_rgb  = self.scale(np.stack((self.ch_int[0],self.zero,self.zero), axis=2), 5)
        self.ch1_rgb  = self.scale(np.stack((self.zero,self.ch_int[1],self.zero), axis=2), 2)

        if plot:
            self.merge_rgb = self.scale(self.ch0_rgb + self.ch1_rgb)

            fig, ax = plt.subplots(figsize=(30,10), nrows=1, ncols=3)
            ax[0].imshow(self.ch0_rgb)
            ax[1].imshow(self.ch1_rgb)
            ax[2].imshow(self.merge_rgb)

            for iax in ax:
                iax.axis("off")

            plt.savefig(plotname, bbox_inches="tight")

    def scale(self, arr, scale=1, thrval=255):

        arr_scale = arr*scale
        arr_scale[arr_scale > thrval] = thrval
        return arr_scale

    def segment_frame(self, pxpmap):

        for i in range(len(self.ch_fns)):
            sgm_img = segment.segment(pxpmap > 0.99, pxpmap > 0.9, pxpmap, min_distance=10)
            wsh_no_artefacts = np.array(segment.correct_artefacts(sgm_img), dtype=int)
            wsh_merged = segment.cell_merge(wsh_no_artefacts, pxpmap, ksz=3, borderlenthr=32, avg_pred_border=0.9)
            self.mask_final[i] = wsh_merged

        # ncolors = 1000
        # colorlist = np.array(plt.cm.tab20b(np.arange(ncolors)/ncolors))
        # np.random.default_rng().shuffle(colorlist)
        # colorlist[0,:] = np.array([0,0,0,1])
        # cm = LinearSegmentedColormap.from_list("shuffled", colorlist, N=ncolors)
        #
        # fig, ax = plt.subplots(figsize=(10,10), nrows=1, ncols=1)
        # ax.imshow(mask_traj[0], cmap=cm, vmin=0, vmax=1000)
        # plt.imshow(self.mask_final)
        # plt.savefig()
        # plt.show()

class img_trajectory(object):

    def __init__(self):
        self.imgs = []

    def add_img(self, img):
        self.imgs.append(img)

    def track(self, ich=0):
        self.mask_loaded = np.where(np.array([(self.imgs[i].mask_final[ich] is not None) for i in np.arange(len(self.imgs))]))[0]
        for ii, iimg in enumerate(self.mask_loaded[1:]):
            print("Processing {:d}/{:d}".format(ii+1, len(self.mask_loaded)), end="\t\t\r")
            self.imgs[iimg].mask_final[ich] = hungarian.correspondence(self.imgs[iimg-1].mask_final[ich], self.imgs[iimg].mask_final[ich])

    def calc_cellstats(self, ich=0):

        self.df_cells = pd.DataFrame(columns=["timepoint", "cell", "intensity", "meantau"])

        self.mask_loaded = np.where(np.array([(self.imgs[i].mask_final[ich] is not None) for i in np.arange(len(self.imgs))]))[0]
        # print("mask_loaded", self.mask_loaded)
        for ii, iimg in enumerate(self.mask_loaded):
            print("Calculating cell stats for image {:d}/{:d}".format(ii+1, len(self.mask_loaded)), end="\t\t\r")
            self.imgs[iimg].ch[ich] = util.SDT(self.imgs[iimg].ch_fns[ich], noload=False)
            self.imgs[iimg].ch_int[ich] = iio.imread(self.imgs[iimg].ch[ich].filename_int_corr_str)
            for j, jcell in enumerate(np.unique(self.imgs[iimg].mask_final[ich])[1:]):
                # print("cell = ", jcell, end=" ")
                jcell_dc = self.imgs[iimg].ch[ich].decay_curve(mask=(self.imgs[iimg].mask_final[ich] == jcell))
                self.df_cells = self.df_cells.append(
                    {
                        "timepoint": ii,
                        "acqtime": self.imgs[iimg].ch[ich].acqtime,
                        "cell": jcell,
                        "npixels": np.sum(self.imgs[iimg].mask_final[ich] == jcell),
                        "intensity": np.mean(self.imgs[iimg].ch_int[ich][self.imgs[iimg].mask_final[ich] == jcell]),
                        "meantau": np.dot(self.imgs[iimg].ch[ich].time(), jcell_dc/np.sum(jcell_dc))
                    }, ignore_index=True
                )

            self.imgs[iimg].ch[ich] = None
            self.imgs[iimg].ch_int[ich] = None

        return self.df_cells

class trajectory_collection(object):

    def __init__(self, ntraj):
        self.list = np.array([img_trajectory() for itraj in range(ntraj)])
