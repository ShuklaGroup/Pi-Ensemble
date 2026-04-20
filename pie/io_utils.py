import typing
from pathlib import Path
from Bio import SeqIO, AlignIO
from biotite.structure.io import load_structure
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.io.pdbx import get_structure as load_cif_structure
from biotite.structure import filter_amino_acids
from biotite.sequence import ProteinSequence
import numpy as np


def read_fasta_first_seq(path: Path) -> str:
    """
    Read the first sequence from a FASTA file using Biopython.
    Returns the sequence as a plain string.
    """
    with open(path, "r") as f:
        records = list(SeqIO.parse(f, "fasta"))
    if not records:
        raise ValueError(f"No sequence found in FASTA file: {path}")
    return str(records[0].seq)


def load_modeled_seq(structure_path: typing.Union[str, Path], chain_id: str) -> str:
    """
    Returns a string with the amino acid sequence of a specific chain
    from a PDB or mmCIF file, including '-' characters for unmodeled residues.

    Assumes residue IDs are sequential integers with no insertion codes.

    Parameters
    ----------
    structure_path : str or Path
        Path to the input structure file (.pdb or .cif).
    chain_id : str
        The chain ID to extract (e.g., "A").

    Returns
    -------
    str
        The amino acid sequence with '-' for missing residues.
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
    chain_atoms = atom_array[atom_array.chain_id == chain_id] # type: ignore
    if chain_atoms.array_length() == 0:
        raise ValueError(f"Chain '{chain_id}' not found in {structure_path}")

    # --- Filter for amino acids ---
    protein_atoms = chain_atoms[filter_amino_acids(chain_atoms)] # type: ignore
    if protein_atoms.array_length() == 0:
        return ""

    # --- Get unique residues ---
    res_ids = protein_atoms.res_id
    res_names = protein_atoms.res_name
    unique_res_ids, indices = np.unique(res_ids, return_index=True) # type: ignore
    unique_res_names = res_names[indices] # type: ignore

    # --- Map residue IDs to 1-letter code ---
    res_map = {
        res_id: ProteinSequence.convert_letter_3to1(res_name)
        for res_id, res_name in zip(unique_res_ids, unique_res_names)
    }

    # --- Fill missing residues with '-' ---
    full_range = range(min(unique_res_ids), max(unique_res_ids) + 1)
    seq = ''.join(res_map.get(i, '-') for i in full_range)

    return seq


def read_alignment_indices(aln_file, ref_seq):
    """
    Read alignment from FASTA file.
    """
    msa = AlignIO.read(aln_file, "fasta")

    # Find the reference sequence in the alignment
    ref_record = next((rec for rec in msa if rec.seq.replace("-", "") == ref_seq), None)
    if ref_record is None:
        raise ValueError("Reference sequence not found in alignment file.")

    ref_aln = str(ref_record.seq)
    aligned_seqs = []
    alignment_indices = []

    for rec in msa:
        seq_aln = str(rec.seq)
        aligned_seqs.append(seq_aln)
        aln_to_seq = []
        seq_pos = 0
        for s in seq_aln:
            if s != '-':
                aln_to_seq.append(seq_pos)
                seq_pos += 1
            else:
                aln_to_seq.append(None)
        alignment_indices.append(aln_to_seq)

    return {'aligned_seqs': aligned_seqs[1:], 'index_maps': alignment_indices[1:]} # Exclude ref
