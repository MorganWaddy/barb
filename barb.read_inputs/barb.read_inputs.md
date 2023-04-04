<a id="barb.read_inputs"></a>

# barb.read\_inputs

<a id="barb.read_inputs.read_in"></a>

#### read\_in

```python
def read_in(jsons)
```

[[view_source]](https://github.com/barb/blob/79d3578bf05a69fa25a4fbb69e564ae26bb4dcfa/barb/read_inputs.py#L6)

Reads data from the command line

**Arguments**:

- `jsons` _[str]_ - input data from json files
  

**Returns**:

- `nFRBs` _[float]_ - number of FRBs detected
- `sensitivity` _[float]_ - sensitivity at FWHM divided by 2
- `R` _[float]_ - telescope beam radius
- `beams` _[float]_ - number of telescope beams
- `tpb` _[float]_ - time per beam
- `flux` _[float]_ - flux measurement of the FRB

