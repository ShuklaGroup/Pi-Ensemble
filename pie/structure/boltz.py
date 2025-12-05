from .base import StructurePredictor
from ..mmseqs_query import query_colabfold
from pathlib import Path
from typing import Union
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


    def _create_input_file(self, sequence: str, outpath: Union[Path, str], **kwargs):
        """
        Generate fasta .yaml file to be used by self._predict. Returns path to file.
        """
        yaml_path = outpath.with_suffix(".yaml")
        msa_mode = kwargs.get("msa_mode", self.msa_mode)
        ligand = kwargs.get("ligand", self.ligand)
        yaml = f"sequences:\n  - protein:\n    id: A\n    sequence: {sequence}\n"

        if msa_mode == "server":
            a3m_path = query_colabfold([sequence], outpath.parent)[0]
            yaml += f"    msa: {a3m_path}\n"
        elif msa_mode == "empty":
            yaml += "    msa: empty\n"
        else:
            raise ValueError(f"{msa_mode} is not a valid MSA mode.")

        if ligand is not None:
            yaml += f"  - ligand:\n    id: B\n    smiles: {ligand}\n"
        
        with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml)
        
        return yaml_path


    def _predict(self, sequence: str, outpath: Union[Path, str], **kwargs):
        outpath = Path(outpath)
        
        output_format = outpath.suffix
        if output_format not in ["pdb", "cif"]:
            raise ValueError("Unknown format for {outpath}. Use .pdb or .cif.")
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

        # Move (copy) files to user-requested location
        shutil.copy(tmp_struct_path, outpath)
        confidence_path = outpath.with_suffix("_confidence.json")
        shutil.copy(tmp_confidence_path, confidence_path)

        prediction = {
            'struct_path': outpath,
            'sequence': sequence,
            'confidence': confidence_path,
        }

        return prediction

