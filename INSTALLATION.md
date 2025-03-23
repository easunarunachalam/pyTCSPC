# Installing `pytcspc`

Create an environment in which to use `pytcspc`. For example, in `conda`, run:
```
conda create -n pytcspc python=3.12
conda activate pytcspc
```

Then install `pytcspc`:
```
pip install pytcspc
```

and verify that it works, e.g. by running:
```
python -c "import pytcspc as pc"
```

It is typical to run `pytcspc` via a Jupyter notebook or in JupyterLab. To install this,
```
pip install jupyterlab ipywidgets
```

It can also be helpful to have a few other packages for actually working with real data: `pip install napari[all] statsmodels statannotations`

# If installing for development
Also install git, then navigate to the local directory where you have cloned the pyTCSPC repository and install:
```
conda install git
cd <cloned_repo_location>
pip install -e .
```
