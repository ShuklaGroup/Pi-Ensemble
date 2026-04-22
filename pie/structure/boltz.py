from .base import StructurePredictor
from ..mmseqs_query import query_colabfold
from pathlib import Path
from typing import List, Union
import shutil
import subprocess


class BoltzPredictor(StructurePredictor):


    def __init__(self, **kwargs):
        """
        Structure prediction object based on ESM3.
        
        Parameters:
            **kwargs: Model-specific parameters.
                device (str): "cpu" or "cuda" (default: "cpu").
                recycling_steps (int): number of recycling steps (default: 1).
                diffusion_samples (int): number of generated samples (default: 1).
                preprocessing_threads (int): number of preprocessing threads (default: 1).
                msa_mode (str): method to obtain MSA ("server" or "empty", default: "server").
                ligand (str or None): fasta string to be attached at the end of input files (may be used to include a ligand).
        """
        # All these params can be overriden in self._predict method.
        self.device = kwargs.get("device", "cpu")
        self.recycling_steps = kwargs.get("recycling_steps", 1)
        self.diffusion_samples = kwargs.get("diffusion_samples", 1)
        self.preprocessing_threads = kwargs.get("preprocessing_threads", 1)
        self.msa_mode = kwargs.get("msa_mode", "server")
        self.ligand = kwargs.get("ligand", None)

        # Check boltz is available
        if shutil.which("boltz") is None:
            raise FileNotFoundError("The command 'boltz' was not found in the current environment.")


    def query_colabfold(self, sequences: list[str], output_dir: Union[Path, str]) -> List[Path]:
        if not sequences:
            raise ValueError("At least one protein sequence is required.")
        return query_colabfold(sequences, output_dir)


    def _create_input_file(self, sequence: str, outpath: Union[Path, str], **kwargs):
        """
        Generate the Boltz YAML input file for a single prediction job.
        """
        outpath = Path(outpath)
        ligand = kwargs.get("ligand", self.ligand)
        msa_mode = kwargs.get("msa_mode", self.msa_mode)
        msa_path = kwargs.get("msa_path")

        if msa_path is None:
            if msa_mode == "server":
                msa_path = self.query_colabfold([sequence], outpath.parent)[0]
            elif msa_mode == "empty":
                msa_path = "empty"
            else:
                raise ValueError(f"{msa_mode} is not a valid MSA mode.")

        yaml_path = outpath.with_suffix(".yaml")
        yaml_lines = [
            "sequences:",
            "  - protein:",
            "    id: A",
            f"    sequence: {sequence}",
            f"    msa: {msa_path}",
        ]

        if ligand is not None:
            yaml_lines.extend(
                [
                    "  - ligand:",
                    "    id: Z",
                    f"    smiles: {ligand}",
                ]
            )

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(line) for line in yaml_lines) + "\n")

        return yaml_path


    def _predict(self, sequence: str, outpath: Union[Path, str], **kwargs):
        outpath = Path(outpath)

        output_format = outpath.suffix.lstrip(".")
        if output_format not in ["pdb", "cif"]:
            raise ValueError(f"Unknown format for {outpath}. Use .pdb or .cif.")
        out_fmt = "mmcif" if output_format == "cif" else "pdb"

        yaml_path = self._create_input_file(sequence, outpath, **kwargs)

        command = [
            'boltz', 'predict', yaml_path,
            '--out_dir', outpath.parent,
            '--accelerator', kwargs.get("device", self.device),
            '--recycling_steps', kwargs.get("recycling_steps", self.recycling_steps),
            '--output_format', out_fmt,
            "--override",
            "--diffusion_samples", kwargs.get("diffusion_samples", self.diffusion_samples),
            "--preprocessing-threads", kwargs.get("preprocessing_threads", self.preprocessing_threads),
        ]

        subprocess.run(command, check=True)

        boltz_outdir = outpath.parent / f"boltz_results_{yaml_path.stem}" / "predictions" / yaml_path.stem
        tmp_struct_path = boltz_outdir / f"{yaml_path.stem}_model_0.{output_format}"
        tmp_confidence_path = boltz_outdir / f"confidence_{yaml_path.stem}_model_0.json"

        shutil.copy(tmp_struct_path, outpath)
        confidence_path = outpath.with_suffix("_confidence.json")
        shutil.copy(tmp_confidence_path, confidence_path)

        prediction = {
            'struct_path': outpath,
            'sequence': sequence,
            'confidence': confidence_path,
        }

        return prediction


    def predict_batch(self, sequences: list[str], outpaths: list[Union[Path, str]], **kwargs):
        if not sequences:
            raise ValueError("At least one protein sequence is required.")

        outpaths = [Path(path) for path in outpaths]
        if len(outpaths) != len(sequences):
            raise ValueError("Number of output paths must match the number of sequences.")

        msa_mode = kwargs.get("msa_mode", self.msa_mode)
        if msa_mode == "server":
            msa_paths = self.query_colabfold(sequences, outpaths[0].parent) # type: ignore
        elif msa_mode == "empty":
            msa_paths = ["empty"] * len(sequences)
        else:
            raise ValueError(f"{msa_mode} is not a valid MSA mode.")

        predictions = []
        for sequence, outpath, msa_path in zip(sequences, outpaths, msa_paths):
            predictions.append(self._predict(sequence, outpath, msa_path=msa_path, **kwargs))

        return predictions
