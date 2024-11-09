__all__ = [
    "SPC",
]

import numpy as np
import os
import sys
from tqdm.notebook import tqdm

from .sdt import get_acqtime
from .sdtfile import _sdt_file as raw_sdtfile
from .util import bits
import xarray as xr


class SPC(object):
    """
    Ref:
    """

    def __init__(
        self,
        filepath,
        pixels_per_line=1,
        nlines=1,
        n_pixels_skip=0,
        n_lines_skip=1,
        show_progress=False,
        max_nframe=1e8,
        nentries=None,
        read_paired_sdt=True,
        save_raw_traj=True,
        save_images=True
    ):
        self.filepath = Path(filepath)
        self.pixels_per_line = pixels_per_line
        self.nlines = nlines
        self.n_pixels_skip = n_pixels_skip
        self.n_lines_skip = n_lines_skip

        if self.filepath.suffix == ".spc":

            if nentries is None:
                nentries = (os.path.getsize(filepath)/4) - 1

                if nentries.is_integer():
                    nentries = int(nentries)
                else:
                    raise ValueError("Error: number of bytes in file not evenly divisible by 4.")
            else:
                # make sure it is an integer
                nentries = int(nentries)

            if read_paired_sdt:
                file = raw_sdtfile( self.get_filepath("sdt") )
                self.acqtime = get_acqtime(file.info)

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

                if show_progress:
                    entry_iterator.close()

                # convert macrotime to units of seconds
                macro *= macro_unit

                # for testing -- if only using the first few events, remove the remaining empty elements
                macro, micro, pixel, line, frame = macro[:ientry], micro[:ientry], pixel[:ientry], line[:ientry], frame[:ientry]

                # valid events are actual photon detection events
                valid = np.logical_not(np.isnan(macro))
                self.macro, self.micro, self.pixel, self.line, self.frame = macro[valid], micro[valid], pixel[valid].astype(int), line[valid].astype(int), frame[valid].astype(int)

            # self.event_coord_list = np.vstack((frame, line, pixel)).T
            # self.event_coord_list = np.vstack((line, pixel)).T

            self.pixel_bins = np.arange(-0.5 + self.n_pixels_skip, self.n_pixels_skip + self.pixels_per_line + 0.5, 1)
            self.line_bins = np.arange(-0.5 + self.n_lines_skip, self.n_lines_skip + self.nlines + 0.5, 1)
            self.frame_bins = np.arange(-0.5, np.max(self.frame) + 1.5, 1)

            self.pixel_linearindex = self.linear_index(self.pixel, self.line)
            self.frame_idxs = np.arange(np.max(self.frame) + 1)

            self.all_photons = xr.Dataset(
                {
                    "microtime": (["time"], self.micro),
                    "frame": (["time"], self.frame),
                    "x": (["time"], self.line),
                    "y": (["time"], self.pixel),
                },
                coords={
                    "time": self.macro
                },
                attrs={
                    # "acqtime": str(self.acqtime),
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
                    "frame time": np.array(frame_start_time)[:(len(self.frame_bins)-1)]
                },
                attrs={
                    # "acqtime": str(self.acqtime),
                    "filepath": str(self.filepath)
                }
            )

            if save_raw_traj:
                raw_traj_path = self.get_filepath(type="nc")
                self.all_photons.to_netcdf(
                    raw_traj_path,
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
                im_path = self.get_filepath(type="nc")
                im_path = im_path.with_stem(im_path.stem + "_im")

                self.images.to_netcdf(
                    im_path,
                    mode="w",
                )
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

    def linear_index(self, pixelvals, linevals):
        return pixelvals + self.pixels_per_line*linevals

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
            use_idxs = event_frame == iframe
            if np.sum(use_idxs) == 0:
                intensity[iframe] = 0
                sum_tau[iframe] = 0
            else:
                intensity[iframe] = np.array(use_idxs, dtype=int).sum()
                sum_tau[iframe] = np.sum(event_micro[use_idxs])

        time = self.frame_idxs*self.dt

        return time[1:-1], intensity[1:-1], sum_tau[1:-1], event_macro, event_micro
