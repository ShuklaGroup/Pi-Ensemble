from .base import SequencePredictor
from ..constants import PMPNN_ALPHABET
from pathlib import Path
from typing import Union
import subprocess
import biotite.structure.io as strucio
import biotite.structure as struc
from Bio import SeqIO
import numpy as np



class ProteinMPNNPredictor(SequencePredictor):

    def __init__(self, pmpnn_path: Union[str, Path]):
        
        self.pmpnn_path = Path(pmpnn_path) / "protein_mpnn_run.py"
        self.alphabet = np.asarray(PMPNN_ALPHABET)
        
        try:
            assert self.pmpnn_path.exists()
        except AssertionError as err:
            print(f"ProteinMPNN prediction script was not found in {self.pmpnn_path}")
            raise err

    def _predict(self, structure: Union[str, Path], outpath: Union[Path, str], **kwargs):
        """
        Predict the protein sequence from a structure.
        
        Parameters:
            structure (str, Path): Path to protein structure.
            **kwargs: Model-specific parameters.
                temperature (float): sampling temperature (default: 0.1).
                seed (int): random seed (default: 1).
                batch_size (int): batch size for ProteinMPNN (defult: 1).
                chain_id (str): chain ID to use (default: "A").
        
        Returns:
            object: A model-specific sequence representation.
        """

        # Gather parameters
        structure = Path(structure)
        outpath = Path(outpath)
        temperature = kwargs.get("temperature", 0.1)
        seed = kwargs.get("seed", 1)
        batch_size = kwargs.get("batch_size", 1)
        chain_id = kwargs.get("chain_id", "A")


        # Check structure format
        if structure.suffix == ".cif":
            structure = self._mmcif_to_pdb(structure, chain_id)
        elif structure.suffix == ".pdb":
            pass
        else:
            raise ValueError(f"Structure format in {structure} not recognized.")

        # Predict sequence
        subprocess.run([
            'python', self.pmpnn_path,
            '--pdb_path', str(structure),
            '--pdb_path_chains', chain_id,
            '--out_folder', outpath,
            '--num_seq_per_target', "1",
            "--sampling_temp", str(temperature),
            "--seed", str(seed),
            "--batch_size", str(batch_size),
            "--save_probs", 1,
            "--pssm_jsonl", "."
        ],
        check=True)

        # Load data
        modeled_seq = self._read_modeled_seq(structure, chain_id)
        fasta_file = outpath / "seqs" / Path(structure.stem).with_suffix(".fa")
        prob_file = outpath / "probs" / Path(structure.stem).with_suffix(".npz")
        npz_data = np.load(prob_file)
        prob_dist_raw = np.squeeze(npz_data["probs"])
        mpnn_mask = np.squeeze(npz_data["mask"]).astype(int)
        prob_dist = prob_dist_raw * mpnn_mask[:, None] # Zero out unmodeled residues

        try:
            assert(len(modeled_seq) == prob_dist.shape[0])
        except AssertionError as err:
            raise err("Sequence and probability distribution have mismatching shapes.")

        sequence = ''.join(
            self.alphabet[np.argmax(p)] if mask else "-"
            for p, mask in zip(prob_dist, mpnn_mask)
        )

        # Package prediction object
        prediction = {
            'fasta_file': fasta_file,
            'prob_file': prob_file,
            'prob_dist': prob_dist,
            'sequence': sequence,
            'modeled_seq': modeled_seq,
        }

        return prediction


    def _mmcif_to_pdb(self, structure: Path, chain_id: str):
        """Convert mmcif to pdb for use with ProteinMPNN.
        """
        cif_struct = strucio.load_structure(structure)
        chain_struct = cif_struct[cif_struct.chain_id == chain_id]
        new_structure = structure.with_suffix(".pdb")
        strucio.save_structure(new_structure, chain_struct)
        return new_structure


    def _read_fasta(self, path: Path):
        """
        Read the first sequence from a FASTA file using Biopython.
        Returns the sequence as a plain string.
        """
        with open(path, "r") as f:
            records = list(SeqIO.parse(f, "fasta"))
        if not records:
            raise ValueError(f"No sequence found in FASTA file: {path}")
        return str(records[0].seq)


    def _read_modeled_seq(structure_path: Union[str, Path], chain_id: str):
        """
        Returns a string with the amino acid sequence of a specific chain
        from a PDB or mmCIF file, including '-' characters for unmodeled residues.

        Assumes residue IDs are sequential integers with no insertion codes.

        Parameters:
            structure_path (str or Path): Path to the input structure file (.pdb or .cif).
            chain_id (str): The chain ID to extract (e.g., "A").

        Returns:
            str: The amino acid sequence with '-' for missing residues.
        """
        structure_path = Path(structure_path)

        # --- Load structure depending on file extension ---
        if structure_path.suffix.lower() == ".cif":
            cif_file = CIFFile.read(str(structure_path))
            atom_array = load_cif_structure(cif_file)
        elif structure_path.suffix.lower() == ".pdb":
            atom_array = load_structure(str(structure_path))
        else:
            raise ValueError(f"Unsupported file type: {structure_path.suffix}")

        # --- Select chain ---
        chain_atoms = atom_array[atom_array.chain_id == chain_id]
        if chain_atoms.array_length() == 0:
            raise ValueError(f"Chain '{chain_id}' not found in {structure_path}")

        # --- Filter for amino acids ---
        protein_atoms = chain_atoms[filter_amino_acids(chain_atoms)]
        if protein_atoms.array_length() == 0:
            return ""

        # --- Get unique residues ---
        res_ids = protein_atoms.res_id
        res_names = protein_atoms.res_name
        unique_res_ids, indices = np.unique(res_ids, return_index=True)
        unique_res_names = res_names[indices]

        # --- Map residue IDs to 1-letter code ---
        res_map = {
            res_id: ProteinSequence.convert_letter_3to1(res_name)
            for res_id, res_name in zip(unique_res_ids, unique_res_names)
        }

        # --- Fill missing residues with '-' ---
        full_range = range(min(unique_res_ids), max(unique_res_ids) + 1)
        seq = ''.join(res_map.get(i, '-') for i in full_range)

        return seq