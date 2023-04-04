<a id="barb.plotting"></a>

# barb.plotting

<a id="barb.plotting.check_likelihood"></a>

#### check\_likelihood

```python
def check_likelihood(bb, lk, save=False)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/plotting.py#L9)

Checks to see if the functions are working

**Arguments**:

- `bb` _float_ - returns numbers spaced evenly w.r.t interval. Similar to a range but instead of step it uses sample number
- `lk` _np.ndarray[float]_ - runs log likelihood function for all instances of b in bb

<a id="barb.plotting.make_hist"></a>

#### make\_hist

```python
def make_hist(data, save=False, output_name="hist_MCMC")
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/plotting.py#L25)

Checks to see if the functions are working

**Arguments**:

- `data` _[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]_ - nFRBs, sensitivity, R, beams, tpb, flux
- `nFRBs` - number of FRBs detected
- `sensitivity` - sensitivity at FWHM divided by 2 (measured in janskys)
- `R` - telescope radius
- `beams` - number of telescope beams
- `tpb` - time per beam
- `flux` - flux measurement of the FRB
- `output_name` _str_ - name of output image

<a id="barb.plotting.make_corner"></a>

#### make\_corner

```python
def make_corner(allsamples, figname="rates_mc.png", save=False)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/plotting.py#L45)

Makes corner plot

**Arguments**:

- `data` _[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]_ - nFRBs, sensitivity, R, beams, tpb, flux
- `nFRBs` - number of FRBs detected
- `sensitivity` - sensitivity at FWHM divided by 2 (measured in janskys)
- `R` - telescope radius
- `beams` - number of telescope beams
- `tpb` - time per beam
- `flux` - flux measurement of the FRB
- `figname` _str_ - name of the output plot

