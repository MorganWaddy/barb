<a id="barb.likelihood"></a>

# barb.likelihood

<a id="barb.likelihood.area"></a>

#### area

```python
def area(R, gamma)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/likelihood.py#L4)

Calculate the beam shape

**Arguments**:

- `R` _float_ - telescope radius
- `gamma` _float_ - Euclidean scaling that is valid for any population with a fixed luminosity distribution, as long as the luminosity does not evolve with redshift and the population has a uniform spatial distribution
  

**Returns**:

- `ar` _float_ - shape of the beam

<a id="barb.likelihood.power_integral"></a>

#### power\_integral

```python
def power_integral(sensitivity, beta)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/likelihood.py#L19)

Calculates the integral of [sensitivity^(beta)]d(sensitivity)

**Arguments**:

- `sensitivity` _float_ - sensitivity at FWHM divided by 2 (measured in janskys)
- `beta` _float_ - Euclidean scaling (gamma +1)
  

**Returns**:

  (sensitivity ** -(beta - 1)) / (beta - 1) (float)

<a id="barb.likelihood.likelihood_list"></a>

#### likelihood\_list

```python
def likelihood_list(vargroup, alpha, beta)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/likelihood.py#L33)

Analyzes all available data to return the likelihood that there will be an FRB

**Arguments**:

- `vargroup` _[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]_ - nFRBs, sensitivity, R, beams, tpb, flux
- `nFRBs` - number of FRBs detected
- `sensitivity` - sensitivity at FWHM divided by 2 (measured in janskys)
- `R` - telescope radius
- `beams` - number of telescope beams
- `tpb` - time per beam
- `flux` - flux measurement of the FRB
- `alpha` _float_ - product of reference rate and gamma
- `beta` _float_ - Euclidean scaling (gamma +1)
  

**Returns**:

- `likelihood_list` _np.ndarray[float]_ - list of likelihoods that there will be an FRB

<a id="barb.likelihood.log_ll"></a>

#### log\_ll

```python
def log_ll(varrest, nFRBs, sensitivity, R, beams, tpb, flux)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/likelihood.py#L86)

Calculates the log of the result from likelihood_list

**Arguments**:

- `nFRBs` - number of FRBs detected
- `sensitivity` - sensitivity at FWHM divided by 2 (measured in janskys)
- `R` - telescope radius
- `beams` - number of telescope beams
- `tpb` - time per beam
- `flux` - flux measurement of the FRB
- `varrest` _float, float_ - alpha, beta
- `alpha` _float_ - product of reference rate and gamma
- `beta` _float_ - Euclidean scaling (gamma +1)
  

**Returns**:

- `log_ll` _np.ndarray[float]_ - log of the list of likelihoods that there will be an FRB

