# pyTCSPC

## 🚧 Experimental -- there will frequently be breaking changes

## 🚧 Please note that environment files are not kept up to date

## Python functions for fluorescence lifetime imaging microscopy (FLIM) and fluorescence correlation spectroscopy (FCS):

### FLIM functions
- read Becker &amp; Hickl .sdt files (based on [`sdtfile`](https://github.com/cgohlke/sdtfile)) into user-friendly `xarray.DataArray`s suitable for further analysis
- produce intensity and lifetime images
- fit decay curves

### FCS functions
- read Becker &amp; Hickl .spc files into user-friendly `xarray.DataArray`s suitable for further analysis
- generate FLIM and intensity images and "videos"
- generate kymographs for line-scanning FCS
