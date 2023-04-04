<a id="barb.mcmc"></a>

# barb.mcmc

<a id="barb.mcmc.sampling"></a>

#### sampling

```python
def sampling(p0,
             vargroup,
             cpu_num,
             nwalkers,
             ndim,
             filename="MCMC_results.h5",
             max_n=100000)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/mcmc.py#L12)

MCMC sampler

**Arguments**:

- `p0` _float_ - a uniform random distribution mimicking the distribution of the data
- `vargroup` _[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]_ - nFRBs, sensitivity, R, beams, tpb, flux
- `nFRBs` - number of FRBs detected
- `sensitivity` - sensitivity at FWHM divided by 2 (measured in janskys)
- `R` - telescope radius
- `beams` - number of telescope beams
- `tpb` - time per beam
- `flux` - flux measurement of the FRB
- `cpu_num` _float_ - number of cpus
  nwalkers (float):walkers are copies of a system evolving towards a minimum
- `ndim` _float_ - number of dimensions to the analysis
- `filename` _str_ - name of the output h5 file
- `max_n` _float_ - maximum number of iterations the MCMC sampler can run
  
  

**Returns**:

- `old_tau` _np.ndarray[float]_ - variable to test convergence

<a id="barb.mcmc.read_samples"></a>

#### read\_samples

```python
def read_samples(filename)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/mcmc.py#L84)

Analyzes output file to compute the samples from the MCMC sampler

**Arguments**:

- `filename` _str_ - name of h5 output file
  

**Returns**:

- `samples` _np.ndarray[float]_ - stored chain of MCMC samples

<a id="barb.mcmc.convert_params"></a>

#### convert\_params

```python
def convert_params(samples)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/mcmc.py#L111)

Converts MCMC samples for the corner plot

**Arguments**:

- `samples` _np.ndarray[float]_ - stored chain of MCMC samples
  

**Returns**:

- `all_samples` _np.ndarray[float]_ - converted chain of MCMC samples

