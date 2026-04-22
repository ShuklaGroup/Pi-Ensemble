from pathlib import Path
from typing import List, Literal, Union
import json
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from .base import StructurePredictor


class ESM3Predictor(StructurePredictor):

    def __init__(self, **kwargs):
        """
        Structure prediction object based on ESM3.
        
        Parameters:
            **kwargs: Model-specific parameters.
                device (str): "cpu" or "cuda" (default: "cpu").
                temperature (float): temperature for structure generation (default: 0.7).
                num_steps (int): number of steps for denoising (default: 8).
        """
        self.device = kwargs.get("device", "cpu")
        try:
            self.model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to(self.device)
        except Exception as err:
            print("ESM3 could not be loaded, ensure weights are available locally.")
            raise err

        self.temperature = kwargs.get("temperature", 0.7)
        self.num_steps = kwargs.get("num_steps", 8)


    def _predict(self, sequence: str, outpath: Union[Path, str], **kwargs):
        """
        Predict the protein structure from a sequence.
        
        Parameters:
            sequence (str): Protein sequence (single-letter amino acids).
            outpath (str, Path): path for predicted structure (e.g., ./output/pred.pdb).
                    
        Returns:
            object: A model-specific structure representation.
        """
        # Gather parameters
        protein = ESMProtein(sequence=sequence)
        outpath = Path(outpath)
        confidence_path = outpath.with_name(f"{outpath.stem}_confidence.json")

        # Predict structure
        protein = self.model.generate(protein, GenerationConfig(track="structure", num_steps=self.num_steps, temperature=self.temperature))
        if outpath.suffix == ".pdb":
            protein.to_pdb(outpath) # type: ignore
        elif outpath.suffix == ".cif":
            protein.to_mmcif(outpath) # type: ignore
        else:
            raise ValueError("Unknown format for {outpath}. Use .pdb or .cif.")

        # Gather confidence metrics
        plddt = protein.plddt.tolist() # type: ignore
        plddt_mean = float(protein.plddt.mean().item()) # type: ignore
        ptm = float(protein.ptm.item()) # type: ignore

        confidence = {
            'plddt': plddt,
            'plddt_mean': plddt_mean,
            'ptm': ptm,
        }

        with open(confidence_path, "w") as outfile:
            json.dump(confidence, outfile, indent=2)

        prediction = {
            'struct_path': outpath,
            'sequence': sequence,
            'confidence': confidence_path,
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
            predictions.append(self._predict(sequence, outpath, **kwargs))

        return predictions
