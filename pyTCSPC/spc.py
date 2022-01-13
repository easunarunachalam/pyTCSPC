from datetime import datetime
import glob
import h5py
import imageio as iio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
from pathlib import Path, PurePath, PureWindowsPath
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve, convolve
import sys
import time
# import xarray as xr


class SPC(object):
    """
    Ref:
    """

    def __init__(
        self,
        filepath,
        pixels_per_line=72,
        nlines=72,
        show_progress=False,
        max_nframe=10,
        read_paired_sdt=True,
        save_raw_traj=True,
        save_images=True
    ):
        self.filepath = filepath
        self.pixels_per_line = pixels_per_line
        self.nlines = nlines

        if filepath.suffix == ".spc":

            nentries = (os.path.getsize(filepath)/4) - 1

            if nentries.is_integer():
                nentries = int(nentries)
            else:
                raise ValueError("Error: number of bytes in file not evenly divisible by 4.")

            if read_paired_sdt:
                sdt = SDT(self.get_filepath("sdt"))
                self.acqtime = sdt.acqtime

            # uncomment to process only the first few entries
            # nentries = 1000000
            macro = np.empty((nentries))
            micro = np.empty((nentries))
            pixel = np.empty((nentries))
            line  = np.empty((nentries))
            frame = np.empty((nentries))

            frame_start_time = [0]

            overflow = ipixel = iline = iframe = 0
            with open(str(filepath), "rb") as f:

                macro_info = list(bits(f.read(4)))
                macro_unit = int("".join(reversed([str(i) for i in macro_info[0:24]])),2) * 1e-10

                entry_iterator = range(nentries)
                if show_progress:
                    entry_iterator = tqdm(entry_iterator, position=0, leave=True, desc="current file")

                for ientry in entry_iterator:

                    # data for single photon, 4 bytes / 2 words
                    photondata = list(bits(f.read(4)))
                    MT      = int("".join(reversed([str(i) for i in photondata[0:12]])),2)
                    ROUT    = list([i for i in photondata[12:16]])

                    if ROUT[0] != 0:
                        ipixel += 1
                    if ROUT[1] != 0:
                        iline += 1
                        ipixel = 0
                    if ROUT[2] != 0:
                        iframe += 1
                        iline = 0
                        ipixel = 0

                        if (max_nframe is not None) and (iframe >= max_nframe):
                            break

                        frame_start_time.append(macro_unit*(MT + overflow))

                    ADC     = int("".join(reversed([str(i) for i in photondata[16:28]])),2)
                    MARK    = photondata[28]
                    GAP     = photondata[29]
                    MTOV    = photondata[30]
                    INVALID = photondata[31]

                    macro[ientry] = MT + overflow
                    micro[ientry] = 4095 - ADC

                    if INVALID == 1:
                        macro[ientry] = micro[ientry] = pixel[ientry] = line[ientry] = frame[ientry] = np.nan
                    else:
                        pixel[ientry], line[ientry], frame[ientry] = ipixel, iline, iframe

                    # print("MT=", MT, "ROUT=", ROUT, "ADC=", ADC, "MARK=", MARK, "GAP=", GAP, "MTOV=", MTOV, "INVALID=", INVALID, overflow, ipixel, iline, iframe, self.macro[ientry])

                    if MTOV == 1: # Macrotime clock overflow
                        if INVALID == 1:
                            if photondata[-4] == 0:
                                n_mtov = int("".join(reversed([str(i) for i in photondata[:28]])),2)
                            else:
                                n_mtov = 1
                            overflow += n_mtov*(2**12)
                            macro[ientry] += n_mtov*(2**12)
                        else:
                            overflow += 2**12
                            macro[ientry] += 2**12

                entry_iterator.close()

                # convert macrotime to units of seconds
                macro *= macro_unit

                # for testing -- if only using the first few events, remove the remaining empty elements
                macro, micro, pixel, line, frame = macro[:ientry], micro[:ientry], pixel[:ientry], line[:ientry], frame[:ientry]

                # valid events are actual photon detection events
                valid = np.logical_not(np.isnan(macro))
                macro, micro, pixel, line, frame = macro[valid], micro[valid], pixel[valid].astype(int), line[valid].astype(int), frame[valid].astype(int)

            # self.event_coord_list = np.vstack((frame, line, pixel)).T
            # self.event_coord_list = np.vstack((line, pixel)).T

            self.pixel_bins = np.arange(-0.5, self.pixels_per_line + 0.5, 1)
            self.line_bins = np.arange(-0.5, self.nlines + 0.5, 1)
            self.frame_bins = np.arange(-0.5, np.max(frame) + 1.5, 1)

            self.pixel_linearindex = self.linear_index(pixel, line)
            self.frame_idxs = np.arange(np.max(frame) + 1)

            self.all_photons = xr.Dataset(
                {
                    "microtime": (["time"], micro),
                    "frame": (["time"], frame),
                    "x": (["time"], line),
                    "y": (["time"], pixel),
                },
                coords={
                    "time": macro
                },
                attrs={
                    "acqtime": str(self.acqtime),
                    "filepath": str(self.filepath),
                    "frame bins": self.frame_bins,
                    "pixel bins": self.pixel_bins,
                    "line bins": self.line_bins,
                }
            )

            self.images = xr.Dataset(
                {
                    "intensity": (["frame time", "x", "y"], self.video(mode="intensity")),
                    "lifetime sum": (["frame time", "x", "y"], self.video(mode="lifetime sum"))
                },
                coords={
                    "frame time": np.array(frame_start_time)
                },
                attrs={
                    "acqtime": str(self.acqtime),
                    "filepath": str(self.filepath)
                }
            )

            if save_raw_traj:
                self.all_photons.to_netcdf(
                    self.get_filepath(type="nc"),
                    mode="w",
                    # mode=self.nc_filewrite_mode(),
                    # engine="h5netcdf",
                    # encoding={
                    #     "microtime": {"zlib": True, "compression": 9},
                    #     "frame": {"zlib": True, "compression": 9},
                    #     "x": {"zlib": True, "compression": 9},
                    #     "y": {"zlib": True, "compression": 9},
                    # }
                )

            if save_images:
                self.images.to_netcdf(
                    self.get_filepath(type="nc"),
                    mode=self.nc_filewrite_mode(),
                    # engine="h5netcdf",
                    # encoding={
                    #     "intensity": {"zlib": True, "compression": 9},
                    #     "lifetime sum": {"zlib": True, "compression": 9},
                    # }
                )


        # elif filepath.suffix == ".h5":
        #
        #     with h5py.File(filepath, "r") as hf:
        #         self.macro                      = np.array(hf.get("macro"))
        #         self.micro                      = np.array(hf.get("micro"))
        #         self.pixel                      = np.array(hf.get("pixel"), dtype=int)
        #         self.line                       = np.array(hf.get("line"), dtype=int)
        #         self.frame                      = np.array(hf.get("frame"), dtype=int)
        #         self.pixels_per_line            = int(np.array(hf.get("pixels_per_line")))
        #         self.nlines                     = int(np.array(hf.get("nlines")))
        #         self.event_coord_list           = np.array(hf.get("event_coord_list"))
        #         self.event_coord_list_framewise = np.array(hf.get("event_coord_list_framewise"))
        #         self.pixel_bins                 = np.array(hf.get("pixel_bins"))
        #         self.line_bins                  = np.array(hf.get("line_bins"))
        #         self.frame_bins                 = np.array(hf.get("frame_bins"))
        #         self.pixel_linearindex          = np.array(hf.get("pixel_linearindex"))
        #         self.frame_idxs                 = np.array(hf.get("frame_idxs"))

        else:
            warnings.warn("Unrecognized file extension.")
            sys.exit()

    def get_filepath(self, type:str="sdt"):

        if type == "sdt":
            return Path.joinpath(self.filepath.parent, self.filepath.stem.replace("_m1", "").replace("_m2", "") + "." + type)
        elif type in ["nc", "h5"]:
            return Path.joinpath(self.filepath.parent, self.filepath.stem + "." + type)
        else:
            raise ValueError("Unrecognized type")

    def nc_filewrite_mode(self):
        """
        Return mode for writing to netcdf4 file -- write ("w") if file does not yet exist, or append ("a") if file already exists
        """
        if self.get_filepath(type="nc").is_file():
            return "a"
        else:
            return "w"

    def event_coord_list_framewise(self):
        return np.vstack(
            (
                self.all_photons["frame"].data,
                self.all_photons["x"].data,
                self.all_photons["y"].data,
            )
        ).T

    # def save_h5(self, fp=None):
    #
    #
    #
    #     with h5py.File(fp, "w") as hf:
    #         hf.create_dataset("macro", data=self.macro)
    #         hf.create_dataset("micro", data=self.micro)
    #         hf.create_dataset("pixel", data=self.pixel)
    #         hf.create_dataset("line", data=self.line)
    #         hf.create_dataset("frame", data=self.frame)
    #         hf.create_dataset("pixels_per_line", data=self.pixels_per_line)
    #         hf.create_dataset("nlines", data=self.nlines)
    #         hf.create_dataset("event_coord_list", data=self.event_coord_list)
    #         hf.create_dataset("event_coord_list_framewise", data=self.event_coord_list_framewise)
    #         hf.create_dataset("pixel_bins", data=self.pixel_bins)
    #         hf.create_dataset("line_bins", data=self.line_bins)
    #         hf.create_dataset("frame_bins", data=self.frame_bins)
    #         hf.create_dataset("pixel_linearindex", data=self.pixel_linearindex)
    #         hf.create_dataset("frame_idxs", data=self.frame_idxs)
    #
    #     return

    def linear_index(self, pixelvals, linevals):
        return pixelvals + self.pixels_per_line*linevals

    # def image(self):
    #     return np.histogramdd(self.event_coord_list, bins=(self.line_bins, self.pixel_bins))[0]

    def video(self, mode="intensity"):
        """

        """
        if mode == "intensity":
            return np.histogramdd(
                self.event_coord_list_framewise(),
                bins=(self.frame_bins, self.line_bins, self.pixel_bins)
            )[0]
        elif mode == "lifetime sum":
            return np.histogramdd(
                self.event_coord_list_framewise(),
                bins=(self.frame_bins, self.line_bins, self.pixel_bins),
                weights=self.all_photons["microtime"].data
            )[0]
        else:
            raise ValueError("Invalid mode")

    def decay_curve(self, roi_pixel_coords, roi_line_coords):

        roi_pixel_linearindex = self.linear_index(roi_pixel_coords, roi_line_coords)
        is_event_in_roi = np.in1d(self.pixel_linearindex, roi_pixel_linearindex)

        microtimes = self.micro[is_event_in_roi]

        return np.histogram(microtimes, bins=np.arange(0,4097,16)-0.5)

    def roi_fli_trajectory(self, roi_pixel_coords, roi_line_coords, dt=0.0155):
        self.dt = dt

        roi_pixel_linearindex = self.linear_index(roi_pixel_coords, roi_line_coords)
        is_event_in_roi = np.in1d(self.pixel_linearindex, roi_pixel_linearindex)

        event_frame, event_macro, event_micro = self.frame[is_event_in_roi], self.macro[is_event_in_roi], self.micro[is_event_in_roi]

        intensity, sum_tau = np.zeros_like(self.frame_idxs), np.zeros_like(self.frame_idxs)
        for iframe in self.frame_idxs:
        #     print(iframe, event_frame[use_idxs])
            use_idxs = event_frame == iframe
            if np.sum(use_idxs) == 0:
                intensity[iframe] = 0
                sum_tau[iframe] = 0
            else:
                intensity[iframe] = np.array(use_idxs, dtype=int).sum()
                sum_tau[iframe] = np.sum(event_micro[use_idxs])

        time = self.frame_idxs*self.dt

        return time[1:-1], intensity[1:-1], sum_tau[1:-1], event_macro, event_micro
