from pathlib import Path
from typing import Union
import json

from .base import StructurePredictor


# Last Hugging Face Hub revisions loadable by the pinned Biohub ESM/transformers stack.
# Later `main` revisions target transformers 5.x and fail to load (ESMFold2) or load
# with randomly initialized weights (ESMC-6B).
DEFAULT_HUB_REVISIONS = {
    "biohub/ESMFold2": "1ebf0e3481a5184eb6171d40615c79e384b48796",
    "biohub/ESMC-6B": "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
}


class ESMFold2Predictor(StructurePredictor):
    def __init__(self, **kwargs):
        """
        Structure prediction object based on ESMFold2.

        Parameters:
            **kwargs: Model-specific parameters.
                model_name (str): Hugging Face model name (default: "biohub/ESMFold2").
                revision (str | None): Hub revision of model_name (default: pinned in DEFAULT_HUB_REVISIONS; None for latest).
                esmc_revision (str | None): Hub revision of the ESMC language model (default: pinned in DEFAULT_HUB_REVISIONS; None for latest).
                device (str): "cpu" or "cuda" (default: "cpu").
                num_loops (int): Number of recycling loops (default: 3).
                num_sampling_steps (int): Number of diffusion sampling steps (default: 50).
                num_diffusion_samples (int): Number of diffusion samples (default: 1).
                seed (int | None): Optional random seed (default: 0).
                protein_id (str): Protein chain ID for single-sequence prediction (default: "A").
        """
        try:
            from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
        except Exception as err:
            raise ImportError(
                "ESMFold2 dependencies could not be imported. Ensure esm, transformers, "
                "torch, and the ESMFold2 model dependencies are installed."
            ) from err

        self.input_builder_cls = ESMFold2InputBuilder
        self.protein_input_cls = ProteinInput

        self.model_name = kwargs.get("model_name", "biohub/ESMFold2")
        self.revision = kwargs.get("revision", DEFAULT_HUB_REVISIONS.get(self.model_name))
        self.esmc_revision = kwargs.get("esmc_revision")
        self._esmc_revision_set = "esmc_revision" in kwargs
        self.device = kwargs.get("device", "cpu")
        self.num_loops = int(kwargs.get("num_loops", 3))
        self.num_sampling_steps = int(kwargs.get("num_sampling_steps", 50))
        self.num_diffusion_samples = int(kwargs.get("num_diffusion_samples", 1))
        self.seed = kwargs.get("seed", 0)
        self.protein_id = kwargs.get("protein_id", "A")

        try:
            from huggingface_hub import snapshot_download

            # load_esmc() takes no revision, so resolve the pinned ESMC snapshot to a local path.
            self.model = ESMFold2Model.from_pretrained(
                self.model_name, revision=self.revision, load_esmc=False
            )
            esmc_id = self.model.config.esmc_id
            if not self._esmc_revision_set:
                self.esmc_revision = DEFAULT_HUB_REVISIONS.get(esmc_id)
            esmc_path = snapshot_download(esmc_id, revision=self.esmc_revision)
            self.model.load_esmc(esmc_path)
            if self.device == "cuda":
                self.model = self.model.cuda() # type: ignore
            elif hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
            self.model = self.model.eval()
        except Exception as err:
            raise RuntimeError(f"ESMFold2 model '{self.model_name}' could not be loaded.") from err

    def _to_float(self, value):
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def _to_list(self, value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _write_structure(self, result, outpath: Path) -> None:
        output_format = outpath.suffix.lower()
        if output_format == ".cif":
            with open(outpath, "w", encoding="utf-8") as outfile:
                outfile.write(result.complex.to_mmcif())
            return

        if output_format == ".pdb":
            tmp_cif_path = outpath.with_name(f"{outpath.stem}_tmp.cif")
            try:
                with open(tmp_cif_path, "w", encoding="utf-8") as outfile:
                    outfile.write(result.complex.to_mmcif())
                self._mmcif_to_pdb(tmp_cif_path, outpath)
            finally:
                tmp_cif_path.unlink(missing_ok=True)
            return

        raise ValueError(f"Unknown format for {outpath}. Use .pdb or .cif.")

    def _mmcif_to_pdb(self, cif_path: Path, pdb_path: Path) -> None:
        try:
            from biotite.structure.io.pdb import PDBFile
            from biotite.structure.io.pdbx import CIFFile, get_structure
        except ImportError as err:
            raise ImportError("Biotite is required to convert ESMFold2 mmCIF output to PDB.") from err

        cif_file = CIFFile.read(str(cif_path))
        structure = get_structure(cif_file, model=1)
        pdb_file = PDBFile()
        pdb_file.set_structure(structure)
        pdb_file.write(str(pdb_path))

    def _build_structure_prediction_input(self, sequence: str, **kwargs):
        from esm.models.esmfold2 import StructurePredictionInput

        sequences = kwargs.get("sequences")
        if sequences is None:
            protein_id = kwargs.get("protein_id", self.protein_id)
            sequences = [self.protein_input_cls(id=protein_id, sequence=sequence)]

        return StructurePredictionInput(sequences=sequences)

    def _predict(self, sequence: str, outpath: Union[Path, str], **kwargs):
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        confidence_path = outpath.with_name(f"{outpath.stem}_confidence.json")

        spi = kwargs.get("structure_prediction_input")
        if spi is None:
            spi = self._build_structure_prediction_input(sequence, **kwargs)

        fold_kwargs = {
            "num_loops": int(kwargs.get("num_loops", self.num_loops)),
            "num_sampling_steps": int(kwargs.get("num_sampling_steps", self.num_sampling_steps)),
            "num_diffusion_samples": int(kwargs.get("num_diffusion_samples", self.num_diffusion_samples)),
            "seed": kwargs.get("seed", self.seed),
        }
        fold_kwargs = {key: value for key, value in fold_kwargs.items() if value is not None}

        result = self.input_builder_cls().fold(self.model, spi, **fold_kwargs)
        self._write_structure(result, outpath)

        plddt = getattr(result, "plddt", None)
        plddt_mean = None
        if plddt is not None:
            plddt_mean_value = plddt.mean() if hasattr(plddt, "mean") else None
            plddt_mean = self._to_float(plddt_mean_value)

        confidence = {
            "plddt": self._to_list(plddt),
            "plddt_mean": plddt_mean,
            "ptm": self._to_float(getattr(result, "ptm", None)),
            "iptm": self._to_float(getattr(result, "iptm", None)),
        }

        with open(confidence_path, "w", encoding="utf-8") as outfile:
            json.dump(confidence, outfile, indent=2)

        prediction = {
            "struct_path": outpath,
            "sequence": sequence,
            "confidence": confidence_path,
        }
        return prediction

    def predict_batch(self, sequences: list[str], outpaths: list[Union[Path, str]], **kwargs):
        if not sequences:
            raise ValueError("At least one protein sequence is required.")

        normalized_outpaths = [Path(path) for path in outpaths]
        if len(normalized_outpaths) != len(sequences):
            raise ValueError("Number of output paths must match the number of sequences.")

        predictions = []
        for sequence, outpath in zip(sequences, normalized_outpaths):
            predictions.append(self.predict(sequence, outpath, **kwargs))

        return predictions
