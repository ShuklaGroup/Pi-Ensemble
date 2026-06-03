# Config examples
These are examples of how to set the config YAML files to use with `run_pie`. Note that a full config file requires three components:

- Structure predictor:
    - [`boltz.yaml`](boltz.yaml)
    - [`esm3.yaml`](esm3.yaml)
    - [`esmfold2.yaml`](esmfold2.yaml)
    - [`bioemu.yaml`](bioemu.yaml) (experimental, does not combine very well with our approach)
- Sequence predictor:
    - [`proteinmpnn.yaml`](proteinmpnn.yaml)
- Interpolation algorithm:
    - [`batch.yaml`](batch.yaml)
    - [`serial.yaml`](serial.yaml)