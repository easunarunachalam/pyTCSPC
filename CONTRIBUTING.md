# Contributing

Contributions are welcome. If you would like to use this package to analyze data collected on non-Bcker&Hickl TCSPC systems, it may be possible to do so. Please take a look at the various packages written by Christoph Gohlke (creator of `sdtfile`); it may be straightforward to use those packages to read data and then do analysis with functions in `pytcspc`.

## building and uploading

Install tools to build and upload:
```
python -m pip install --upgrade build
python -m pip install --upgrade twine
```

Build the wheel:
```
python -m build
```

Test local installation:
```
pip install .\dist\pytcspc-1.0.0-py2.py3-none-any.whl --force-reinstall
```

Upload wheel to PyPI:
```
python -m twine upload --repository testpypi dist/*
```