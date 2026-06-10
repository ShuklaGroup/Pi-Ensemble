# Pi-Ensemble
[PREPRINT PLACEHOLDER]
[DataBank PLACEHOLDER]


Pi-Ensemble is a modular framework for generating protein sequence/structure interpolations between two template structures. The package combines:

- a **structure predictor** that maps sequence -> structure
- an **inverse folding model** that maps structure -> sequence probabilities
- an **interpolation algorithm** that mixes information between two templates

The code is organized under [`pie/`](pie/), with the main entrypoint in [`pie/run_pie.py`](pie/run_pie.py). Scientific context for the method will be linked here once the associated manuscript and data resources are public.

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

Pi-Ensemble runs pre-trained models for structure prediction and sequence design. The implemented models and sample configuration files are listed in the tables below.

### Structure Prediction <a name="available-structure"></a>

| Registry name | Class | Example config | Model reference |
| --- | --- | --- | --- |
| `esm3` | [`ESM3Predictor`](pie/structure/esm3.py) | [`examples/esm3.yaml`](examples/esm3.yaml) | [ESM3](https://www.science.org/doi/10.1126/science.ads0018) |
| `boltz` | [`BoltzPredictor`](pie/structure/boltz.py) | [`examples/boltz.yaml`](examples/boltz.yaml) | [Boltz](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v4) |
| `bioemu` | [`BioEmuPredictor`](pie/structure/bioemu.py) | [`examples/bioemu.yaml`](examples/bioemu.yaml) | [BioEmu](https://www.science.org/doi/10.1126/science.adv9817) |
| `ESMFold2` | [`ESMFold2Predictor`](pie/structure/esmfold2.py) | [`examples/esmfold2.yaml`](examples/esmfold2.yaml) | [ESMFold2](https://biohub.ai/papers/esm_protein.pdf) |

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

Please note that installation and usage configuration may vary according to the models you wish to use.

### Installation <a name="install"></a>

#### Docker

Docker is the recommended path when you want the tested dependency layout without reproducing the host setup manually. The repository ships a CUDA 12.8 image definition and Compose file:

- [`Dockerfile`](Dockerfile)
- [`docker-compose.yml`](docker-compose.yml)

The published image is `zcorn/pi-ensemble:cuda12.8`. Pull it with:

```bash
docker pull zcorn/pi-ensemble:cuda12.8
```

Or build it locally with:

```bash
docker compose build PI-Ensemble
```

Run Pi-Ensemble inside the container with:

```bash
docker compose run --rm PI-Ensemble run_pie test/48G7g/boltz_proteinmpnn_serial.yaml
```

The Compose setup mounts the repository at `/workspace`, keeps Hugging Face downloads in the host cache, persists model caches in a named Docker volume, and reserves all available NVIDIA GPUs. The main `pie` environment uses Python 3.12 for ESMFold2 support. BioEmu is installed in a separate Python 3.11 `bioemu` Conda environment inside the image to avoid dependency conflicts with Boltz, and cg2all is kept in its own Python 3.11 environment.

If you use ESM3, authenticate with Hugging Face before the first run because the model repository is gated:

```bash
huggingface-cli login
```

#### Host Installation

For host installation, use [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview). Pi-Ensemble uses separate environments because Boltz and BioEmu require incompatible dependency sets.

Create the main Pi-Ensemble environment. This environment uses Python 3.12 because Biohub ESMFold2 requires it:

```bash
conda env create -f environment.yml
conda activate pie
```

Clone ProteinMPNN at a known location:

```bash
git clone https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN
```

If `/opt` is not writable on your machine, clone elsewhere and set `sequence_prediction.kwargs.pmpnn_path` in your YAML config to that directory.

Create the optional BioEmu environment when you need `structure_prediction.model: "bioemu"`:

```bash
conda env create -f environment-bioemu.yml
```

Create the optional cg2all environment when you enable `cg2all: true`:

```bash
conda env create -f environment-cg2all.yml
```

The provided cg2all environment is CPU-only. Pi-Ensemble always runs cg2all with `--device cpu`.

The host environment files mirror the container layout:

- [`environment.yml`](environment.yml): main Python 3.12 `pie` environment with Boltz, Biohub ESM/ESMFold2, OpenMM, and the Pi-Ensemble package
- [`environment-bioemu.yml`](environment-bioemu.yml): separate Python 3.11 `bioemu` environment
- [`environment-cg2all.yml`](environment-cg2all.yml): separate Python 3.11 `cg2all` environment

ESMFold2 dependencies are installed from Biohub's GitHub ESM package rather than the generic PyPI `esm` package.

When running BioEmu from a host install, the default configuration assumes the auxiliary environment is named `bioemu`. Override `bioemu_environment` in the model kwargs only if you use another name.


### Prediction <a name="prediction"></a>

Pi-Ensemble is executed from a YAML configuration file through [`pie/run_pie.py`](pie/run_pie.py). A run consists of three blocks:

- `structure_prediction`: selects the sequence -> structure model
- `sequence_prediction`: selects the structure -> sequence model
- `interpolation`: selects the interpolation algorithm and its settings

After installing the package, run:

```bash
run_pie path/to/config.yaml
```

Example:

```bash
run_pie test/48G7g/esm3_proteinmpnn_serial.yaml
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

OpenMM minimization uses `minimize_forcefield`, which defaults to `charmm36_2024.xml`. You can override it with any [OpenMM-compatible forcefield](https://ommprotocol.readthedocs.io/en/latest/forcefields.html) XML name, or with a list when multiple XML files are needed:

```yaml
interpolation:
  kwargs:
    minimize: true
    minimize_forcefield: "amber14-all.xml"
```

```yaml
interpolation:
  kwargs:
    minimize: true
    minimize_forcefield:
      - "amber14/protein.ff14SB.xml"
      - "amber14/tip3pfb.xml"
```

### Visualization <a name="visualization"></a>

Visualization scripts will be shared soon.

## Citations <a name="citations"></a>

If you use Pi-Ensemble, please cite its associated publication as well as the models called by your config choices.

[PLACEHOLDER FOR Pi-Ensemble CITATION]

| If your run used... | Also cite... |
| --- | --- |
| `esm3` | [ESM3](https://www.science.org/doi/10.1126/science.ads0018) |
| `esmfold2` | [ESMFold2](https://biohub.ai/papers/esm_protein.pdf) |
| `boltz` | [Boltz](https://www.biorxiv.org/content/10.1101/2024.11.19.624167v4) |
| `bioemu` | [BioEmu](https://www.science.org/doi/10.1126/science.adv9817) |
| `proteinmpnn` | [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) |
| `cg2all` | [cg2all](https://www.sciencedirect.com/science/article/pii/S0969212623004458) |
| `minimize` | [OpenMM](https://pubs.acs.org/doi/10.1021/acs.jpcb.3c06662) |

## License <a name="license"></a>

Pi-Ensemble itself is distributed under the [MIT License](LICENSE). Model outputs and pretrained weights may be subject to additional third-party licenses or usage restrictions, depending on which options you choose. Users are responsible for obtaining any required model weights and complying with the corresponding terms of use.

| Component | License / source note |
| --- | --- |
| ESM3/ESMFold2 | [MIT License](https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md) |
| Boltz | [MIT License](https://github.com/jwohlwend/boltz/blob/main/LICENSE) |
| BioEmu | [MIT License](https://github.com/microsoft/bioemu/blob/main/LICENSE) |
| ProteinMPNN | [MIT License](https://github.com/dauparas/ProteinMPNN/blob/main/LICENSE)  |
| cg2all | [Apache License 2.0](https://github.com/huhlim/cg2all/blob/main/LICENSE)  |
| OpenMM | Various open [licenses](https://github.com/openmm/openmm/blob/master/docs-source/licenses/Licenses.txt) |

## Generative AI Acknowledgement <a name="gen-ai-ackn"></a>

This repository was built with input from [Codex](https://openai.com/codex/).
