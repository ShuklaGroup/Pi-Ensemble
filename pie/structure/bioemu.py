from pathlib import Path
from typing import List, Union

from .base import StructurePredictor
from ..mmseqs_query import query_colabfold

from bioemu.sample import main as sample


class BioEmuPredictor(StructurePredictor):
    def __init__(self, **kwargs):
        """
        Structure prediction object based on BioEmu.

        Parameters:
            **kwargs: Model-specific parameters.
                num_samples (int): Number of ensemble samples to generate (default: 1).
                sample_index (int): Which generated PDB to expose as struct_path (default: 0).
                batch_size_100 (int): BioEmu batch-size heuristic parameter (default: 10).
                model_name (str | None): Pretrained BioEmu model name (default: "bioemu-v1.1").
                ckpt_path (str | Path | None): Optional checkpoint path.
                model_config_path (str | Path | None): Optional model config path.
                denoiser_type (str | None): Denoiser type (default: "dpm").
                denoiser_config (str | Path | None): Optional denoiser config path.
                cache_embeds_dir (str | Path | None): Optional embeddings cache directory.
                cache_so3_dir (str | Path | None): Optional SO3 cache directory.
                msa_host_url (str | None): Optional MSA server URL.
                filter_samples (bool): Whether to filter unphysical samples (default: True).
                steering_config (str | Path | None): Optional steering config.
                base_seed (int | None): Optional sampling seed.
        """
        self.num_samples = int(kwargs.get("num_samples", 1))
        self.sample_index = int(kwargs.get("sample_index", 0))
        self.batch_size_100 = int(kwargs.get("batch_size_100", 10))
        self.model_name = kwargs.get("model_name", "bioemu-v1.1")
        self.ckpt_path = kwargs.get("ckpt_path")
        self.model_config_path = kwargs.get("model_config_path")
        self.denoiser_type = kwargs.get("denoiser_type", "dpm")
        self.denoiser_config = kwargs.get("denoiser_config")
        self.cache_embeds_dir = kwargs.get("cache_embeds_dir")
        self.cache_so3_dir = kwargs.get("cache_so3_dir")
        self.msa_host_url = kwargs.get("msa_host_url")
        self.filter_samples = bool(kwargs.get("filter_samples", True))
        self.steering_config = kwargs.get("steering_config")
        self.base_seed = kwargs.get("base_seed")

        if sample is None:
            raise ImportError("bioemu is not available in the current environment.")

    def query_colabfold(self, sequences: list[str], output_dir: Union[Path, str]) -> List[Path]:
        if not sequences:
            raise ValueError("At least one protein sequence is required.")
        return query_colabfold(sequences, output_dir)

    def _sample(self, msa_path: Union[str, Path], output_dir: Path, **kwargs) -> None:
        num_samples = int(kwargs.get("num_samples", self.num_samples))
        sample_kwargs = {
            "batch_size_100": int(kwargs.get("batch_size_100", self.batch_size_100)),
            "model_name": kwargs.get("model_name", self.model_name),
            "ckpt_path": kwargs.get("ckpt_path", self.ckpt_path),
            "model_config_path": kwargs.get("model_config_path", self.model_config_path),
            "denoiser_type": kwargs.get("denoiser_type", self.denoiser_type),
            "denoiser_config": kwargs.get("denoiser_config", self.denoiser_config),
            "cache_embeds_dir": kwargs.get("cache_embeds_dir", self.cache_embeds_dir),
            "cache_so3_dir": kwargs.get("cache_so3_dir", self.cache_so3_dir),
            "msa_host_url": kwargs.get("msa_host_url", self.msa_host_url),
            "filter_samples": bool(kwargs.get("filter_samples", self.filter_samples)),
            "steering_config": kwargs.get("steering_config", self.steering_config),
            "base_seed": kwargs.get("base_seed", self.base_seed),
        }
        sample_kwargs = {key: value for key, value in sample_kwargs.items() if value is not None}
        sample(str(msa_path), num_samples, output_dir, **sample_kwargs) # type: ignore[misc]

    def _find_sample_files(self, output_dir: Path) -> tuple[Path, Path]:
        topology_files = sorted(output_dir.rglob("topology.pdb"))
        xtc_files = sorted(output_dir.rglob("samples.xtc"))

        if not topology_files:
            raise FileNotFoundError(f"Could not find topology.pdb under {output_dir}")
        if not xtc_files:
            raise FileNotFoundError(f"Could not find samples.xtc under {output_dir}")

        return topology_files[0], xtc_files[0]

    def _extract_sample_pdb(self, topology_path: Path, xtc_path: Path, sample_index: int, outpath: Path) -> None:
        try:
            import mdtraj as md
        except ImportError as err:
            raise ImportError(
                "mdtraj is required to extract a BioEmu sample frame from samples.xtc."
            ) from err

        traj = md.load_xtc(str(xtc_path), top=str(topology_path))
        if traj.n_frames == 0:
            raise ValueError(f"No frames found in {xtc_path}")
        if sample_index >= traj.n_frames:
            raise IndexError(
                f"Requested sample_index {sample_index} but only found {traj.n_frames} frames in {xtc_path}."
            )

        traj[sample_index].save_pdb(str(outpath))

    def _predict(self, sequence: str, outpath: Union[Path, str], **kwargs):
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        msa_path = kwargs.pop("msa_path", None)
        if msa_path is None:
            msa_path = self.query_colabfold([sequence], outpath.parent)[0]

        bioemu_output_dir = outpath.parent / f"bioemu_{outpath.stem}"
        bioemu_output_dir.mkdir(parents=True, exist_ok=True)

        self._sample(msa_path, bioemu_output_dir, **kwargs)

        sample_index = int(kwargs.get("sample_index", self.sample_index))
        topology_path, xtc_path = self._find_sample_files(bioemu_output_dir)
        self._extract_sample_pdb(topology_path, xtc_path, sample_index, outpath)

        prediction = {
            "struct_path": outpath,
            "sequence": sequence,
            "ensemble_dir": bioemu_output_dir,
            "topology": topology_path,
            "trajectory": xtc_path,
            "sample_index": sample_index,
        }
        return prediction

    def predict_batch(self, sequences: list[str], outpaths: list[Union[Path, str]], **kwargs):
        if not sequences:
            raise ValueError("At least one protein sequence is required.")

        normalized_outpaths = [Path(path) for path in outpaths]
        if len(normalized_outpaths) != len(sequences):
            raise ValueError("Number of output paths must match the number of sequences.")

        msa_paths = self.query_colabfold(sequences, normalized_outpaths[0].parent)
        predictions = []
        for sequence, outpath, msa_path in zip(sequences, normalized_outpaths, msa_paths):
            predictions.append(self.predict(sequence, outpath, msa_path=msa_path, **kwargs))
        return predictions
