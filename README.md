# PI-Ensemble
[PREPRINT PLACEHOLDER]
[DataBank PLACEHOLDER]


$\pi$-Ensemble is a modular framework for generating protein sequence/structure interpolations between two template structures. The package combines:

- a **structure predictor** that maps sequence -> structure
- an **inverse folding model** that maps structure -> sequence probabilities
- an **interpolation algorithm** that mixes information between two templates

The code is organized under [`pie/`](/media/diego/datapartition/21-InterpolatedEnsemble/PIE_new/pie), with the main entrypoint in [`pie/run_pie.py`](/media/diego/datapartition/21-InterpolatedEnsemble/PIE_new/pie/run_pie.py). Scientific context for the method will be linked here once the associated manuscript and data resources are public.

[FRONTPAGE IMAGE]

<details close><summary><b>Table of contents</b></summary>

- [Implemented Models](#available)
  - [Structure Prediction](#available-structure)
  - [Inverse Folding](#available-sequence)
  - [Interpolation Algorithms](#available-algorithm)
- [Usage](#usage)
  - [Installation](#install)
  - [Prediction](#prediction)
  - [Output Interpretation](#output)
  - [Visualization](#visualization)
- [Citations](#citations)
- [License](#license)
</details> 

## Implemented Models <a name="available"></a>

$\pi$-Ensemble runs pre-trained models for structure prediction and sequence design. The implemented models and sample configuration files are listed in the tables below.

### Structure Prediction <a name="available-structure"></a>

| Registry name | Class | Example config | Model reference |
| --- | --- | --- | --- |
| `esm3` | [`ESM3Predictor`](pie/structure/esm3.py) | [`examples/esm3.yaml`](examples/esm3.yaml) | [ESM3](https://www.science.org/doi/10.1126/science.ads0018) |
| `boltz` | [`BoltzPredictor`](pie/structure/boltz.py) | [`examples/boltz.yaml`](examples/boltz.yaml) | [Boltz](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v4) |
| `bioemu` | [`BioEmuPredictor`](pie/structure/bioemu.py) | [`examples/bioemu.yaml`](examples/bioemu.yaml) | [BioEmu](https://www.science.org/doi/10.1126/science.adv9817) |

### Inverse Folding <a name="available-sequence"></a>

| Registry name | Class | Example config | Model reference |
| --- | --- | --- | --- |
| `proteinmpnn` | [`ProteinMPNNPredictor`](pie/sequence/proteinmpnn.py) | [`examples/proteinmpnn.yaml`](examples/proteinmpnn.yaml) | [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) |

### Interpolation Algorithms <a name="available-algorithm"></a>

| Registry name | Class | Example config |
| --- | --- | --- |
| `serial` | [`SerialInterpolation`](pie/algorithms/serial.py) | [examples/serial.yaml](examples/serial.yaml) |
| `batch` | [`BatchInterpolation`](pie/algorithms/batch.py) | [examples/batch.yaml](examples/batch.yaml) |

## Usage <a name="usage"></a>

Installation and usage configuration vary according to the models you wish to use.

### Installation <a name="install"></a>

[INSTALLATION INSTRUCTIONS HERE]

### Prediction <a name="prediction"></a>

$\pi$-Ensemble is executed from a YAML configuration file through [`pie/run_pie.py`](pie/run_pie.py). A run consists of three blocks:

- `structure_prediction`: selects the sequence -> structure model
- `sequence_prediction`: selects the structure -> sequence model
- `interpolation`: selects the interpolation algorithm and its settings

After installing the package, run:

```bash
run_pie path/to/config.yaml
```

Example:

```bash
run_pie examples/esm3_proteinmpnn_serial.yaml
```

The config schema is:

```yaml
structure_prediction:
  model: "<registry name>"
  kwargs:
    ...

sequence_prediction:
  model: "<registry name>"
  kwargs:
    ...

interpolation:
  name: "<registry name>"
  kwargs:
    ...
```

The registry names currently available are found in the tables above and in the following files:

- [`pie/structure/registry.py`](pie/structure/registry.py)
- [`pie/sequence/registry.py`](pie/sequence/registry.py)
- [`pie/algorithms/registry.py`](pie/algorithms/registry.py)

### Output Interpretation <a name="output"></a>

Outputs are written under the `outpath` specified in the interpolation config.

For the `serial` algorithm, outputs are organized by interpolation weight, direction, and round:

```text
outpath/
  round_0/
  weight_<value>/
    direction_A/
      round_1/
      round_2/
      ...
    direction_B/
      round_1/
      round_2/
      ...
  log.json
```

For the `batch` algorithm, outputs are organized by round, direction, and generated sequence index:

```text
outpath/
  round_0/
  round_1/
    direction_A/
      sequence_000/
      sequence_001/
      ...
    direction_B/
      sequence_000/
      sequence_001/
      ...
  round_2/
  ...
  log.json
```

Common output artifacts include:

- predicted structure files such as `structure.pdb`
- model-specific auxiliary files, for example confidence JSON files
- inverse-folding outputs such as `seqs/*.fa` and `probs/*.npz`
- `log.json`, a lightweight session summary of the run

If enabled by the interpolation config, postprocessing outputs are also written at the end of the run:

- `cg2all/` for backbone-to-all-atom conversion
- `minimized/` for OpenMM-minimized structures

### Visualization <a name="visualization"></a>

Visualization scripts will be shared soon.

## Citations <a name="citations"></a>

If you use $\pi$-Ensemble, please cite its associated publication as well as the models called by your config choices.

[PLACEHOLDER FOR PI-ENSEMBLE CITATION]

| If your run used... | Also cite... |
| --- | --- |
| `esm3` | [ESM3](https://www.science.org/doi/10.1126/science.ads0018) |
| `boltz` | [Boltz](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v4) |
| `bioemu` | [BioEmu](https://www.science.org/doi/10.1126/science.adv9817) |
| `proteinmpnn` | [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) |

## License <a name="license"></a>

$\pi$-Ensemble itself is distributed under the [MIT License](LICENSE). Model outputs and pretrained weights may be subject to additional third-party licenses or usage restrictions, depending on which options you choose. Users are responsible for obtaining any required model weights and complying with the corresponding terms of use.

| Component | License / source note |
| --- | --- |
| ESM3 | See details [here](https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md) |
| Boltz | [MIT License](https://github.com/jwohlwend/boltz/blob/main/LICENSE) |
| BioEmu | [MIT License](https://github.com/microsoft/bioemu/blob/main/LICENSE) |
| ProteinMPNN | [MIT License](https://github.com/dauparas/ProteinMPNN/blob/main/LICENSE)  |
