# pyTCSPC

🚧 **Experimental -- there will frequently be breaking changes**

## a Python library for fluorescence lifetime imaging microscopy (FLIM) and fluorescence correlation spectroscopy (FCS) data analysis:
### FLIM functions
- read Becker &amp; Hickl .sdt files (based on [`sdtfile`](https://github.com/cgohlke/sdtfile)) into user-friendly `xarray.DataArray`s suitable for further analysis
- produce intensity and lifetime images
- fit decay curves

### FCS functions
- read Becker &amp; Hickl .spc files into user-friendly `xarray.DataArray`s suitable for further analysis
- generate FLIM and intensity images and "videos"
- generate kymographs for line-scanning FCS
- calculate correlation functions (based on [`multipletau`](https://github.com/FCS-analysis/multipletau))

## environments
- generate yaml files by `conda env export --no-builds > mmpy.yml`
