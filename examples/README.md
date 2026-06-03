# Config examples
These are examples of how to set the config YAML files to use with `run_pie`. Note that a full config file requires three components:

- Structure predictor:
    - [`boltz.yaml`](examples/boltz.yaml)
    - [`esm3.yaml`](examples/esm3.yaml)
    - [`esmfold2.yaml`](examples/esmfold2.yaml)
    - [`bioemu.yaml`](examples/bioemu.yaml) (experimental, does not combine very well with our approach)
- Sequence predictor:
    - [`proteinmpnn.yaml`](examples/proteinmpnn.yaml)
- Interpolation algorithm:
    - [`batch.yaml`](examples/batch.yaml)
    - [`serial.yaml`](examples/serial.yaml)