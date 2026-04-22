from __future__ import annotations

import numpy as np
from typing import Dict, List, Union
from pathlib import Path
import tqdm
import subprocess
from openmm.app import ForceField, PDBFile, Simulation, NoCutoff
from openmm import LangevinIntegrator
from openmm.unit import kelvin, picosecond # type: ignore
from .base import InterpolationAlgorithm
from ..session.session import SessionTracker
from ..constants import ONE_TO_THREE
from ..structure.base import StructurePredictor
from ..sequence.base import SequencePredictor
from ..io_utils import read_alignment_indices
from ..alignment_utils import compute_alignment_indices



class SerialInterpolation(InterpolationAlgorithm):

    def __init__(
        self,
        ref_sequence: str,
        number_steps: int, 
        weight_start: float,
        weight_end: float,
        weight_step: float,
        structure_model: StructurePredictor, 
        sequence_model: SequencePredictor,
        template_1: Path,
        chain_id_1: str,
        template_2: Path,
        chain_id_2: str,
        outpath: Path,
        cg2all: bool = False,
        minimize: bool = False,
        **kwargs
    ):

        self.ref_sequence = ref_sequence
        self.number_steps = number_steps
        if weight_step <= 0:
            raise ValueError("weight_step must be positive.")
        if weight_end < weight_start:
            raise ValueError("weight_end must be greater than or equal to weight_start.")
        self.mixing_weights = np.arange(weight_start, weight_end + (weight_step / 2.0), weight_step)
        self.structure_model = structure_model
        self.sequence_model = sequence_model
        self.outpath = Path(outpath)
        
        self.cg2all = cg2all
        self.minimize = minimize

        if self.cg2all:
            self.cg2all_environment = kwargs.get("cg2all_environment", "cg2all")
            self.cg2all_device = kwargs.get("cg2all_device", "cpu")
            if self.minimize:
                self.forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        
        self.template_1 = self._process_template(template_1, chain_id_1)
        self.template_2 = self._process_template(template_2, chain_id_2)

        self.aln_file = kwargs.get("aln_file", None)

        # self.alignment_map : Dict['aligned_seqs' : List[str], 'index_maps' : List[List[int]]]
        if self.aln_file is not None:
            self.alignment_map = read_alignment_indices(self.aln_file, self.ref_sequence)
        else:
            seqs = [t['modeled_seq'] for t in (self.template_1, self.template_2)]
            self.alignment_map = compute_alignment_indices(seqs, self.ref_sequence)

        self._align_prob_dist()

        self.logger = SessionTracker()
        self.logger.record("templates", "template_1", data=self._serializable_record(self.template_1))
        self.logger.record("templates", "template_2", data=self._serializable_record(self.template_2))


    def _process_template(self, template: Path, chain_id: str):
        
        template_dict = {
            'struct_path': template,
            'chain_id': chain_id,
        }

        outpath = self.outpath / "round_0"
        outpath.mkdir(parents=True, exist_ok=True)
        seq_pred = self.predict_sequence({'struct_path': template}, outpath, chain_id=chain_id)
        template_dict.update(seq_pred)

        return template_dict


    def _align_prob_dist(self):
        """Rewrites probability distribution according to alignment.
        """
        aligned_seqs = self.alignment_map['aligned_seqs']
        index_maps = self.alignment_map['index_maps']

        self.template_1['aligned_seq'] = aligned_seqs[0]
        self.template_2['aligned_seq'] = aligned_seqs[1]

        self.template_1['index_map'] = index_maps[0]
        self.template_2['index_map'] = index_maps[1]

        total_length = len(aligned_seqs[0])
        alphabet_size = len(self.sequence_model.alphabet) # type: ignore
        for temp in (self.template_1, self.template_2):
            new_dist = np.zeros((total_length, alphabet_size))
            for i in range(total_length):
                mapped = temp['index_map'][i]
                if mapped is not None:
                    new_dist[i] = temp['prob_dist'][mapped]
            temp['prob_dist'] = new_dist

        assert(self.template_1['prob_dist'].shape == self.template_2['prob_dist'].shape)


    def predict_sequence(self, structure: dict, outpath: Union[str, Path], **kwargs):
        return self.sequence_model.predict(structure['struct_path'], outpath, **kwargs) # type: ignore


    def predict_structure(self, sequence: str, outpath: Union[str, Path], **kwargs):
        return self.structure_model.predict(sequence, outpath, **kwargs) # type: ignore


    def mix_probabilities(self, seq_1: dict, seq_2: dict, weight: float, **kwargs):
        
        prob_dist = weight*seq_1['prob_dist'] + (1-weight)*seq_2['prob_dist']

        sequence = ''.join(
            self.sequence_model.alphabet[np.argmax(p)]  # type: ignore
            for p in prob_dist
        )

        return sequence


    def run(self):

        generated_structs = []

        for weight in self.mixing_weights:
            weight_label = str(float(weight))
            for direction in ['A', 'B']:
                direction_outpath = self.outpath / f"weight_{weight_label}" / f"direction_{direction}"

                if direction == 'A':
                    anchor = self.template_1
                    mobile = self.template_2
                else:
                    anchor = self.template_2
                    mobile = self.template_1

                for step in range(1, self.number_steps + 1):
                    step_outpath = direction_outpath / f"round_{repr(step)}"
                    step_outpath.mkdir(parents=True, exist_ok=True)

                    # Mix probs
                    new_seq = self.mix_probabilities(anchor, mobile, weight)

                    # Predict structure
                    struct_path = step_outpath / "structure.pdb"
                    new_struct = self.predict_structure(new_seq, struct_path)
                    generated_structs.append(new_struct)

                    # Predict sequence
                    mobile = self.predict_sequence(new_struct, step_outpath)

                    record = {
                        **new_struct,
                        **mobile,
                        "weight": weight,
                        "direction": direction,
                        "step": step,
                        "source_anchor": str(anchor["struct_path"]),
                    }
                    self.logger.record(
                        f"weight_{weight_label}",
                        f"direction_{direction}",
                        f"round_{repr(step)}",
                        data=self._serializable_record(record),
                    )

                self.logger.save_json(self.outpath / "log.json")

        if self.cg2all:
            cg2all_outpath = self.outpath / "cg2all"
            for struct in generated_structs:
                self.extract_backbone_coords_to_pdb(struct, cg2all_outpath)
            self.run_cg2all(cg2all_outpath)

            if self.minimize:
                min_outpath = self.outpath / "minimized"
                self.run_minimization(cg2all_outpath, min_outpath)


    def _serializable_record(self, record: Dict[str, object]) -> Dict[str, object]:
        serialized: Dict[str, object] = {}
        for key, value in record.items():
            if key == "prob_dist":
                continue
            if isinstance(value, Path):
                serialized[key] = str(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized


    def extract_backbone_coords_to_pdb(self, structure: dict, outpath: Path):
        """
        Extracts backbone atoms from a PDB and updates residue names to match ref_sequence.

        Args:
            structure (dict): Object with struct_path.
            outpath (Path): Save dir for backbone PDB file.

        Returns:
            Path: Path to the new backbone-only PDB file.
        """
        keep = {"N", "CA", "C", "O"}
        in_path = Path(structure["struct_path"])
        outfile = outpath / (in_path.stem + "_backbone.pdb")
        outfile.parent.mkdir(parents=True, exist_ok=True)

        residue_index = -1  # Starts before first residue
        atom_count = 0

        with open(in_path, "r") as fin, open(outfile, "w") as fout:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() in keep:
                    atom_name = line[12:16].strip()

                    # Start of new residue: assume every 4 backbone atoms is one residue
                    if atom_name == "N":
                        residue_index += 1
                        if residue_index >= len(self.ref_sequence):
                            raise ValueError(f"Too many residues in PDB file for given reference sequence of length {len(self.ref_sequence)}.")

                    # Replace residue name
                    if residue_index < len(self.ref_sequence):
                        new_resname = ONE_TO_THREE[self.ref_sequence[residue_index]]
                        line = line[:17] + f"{new_resname:>3}" + line[20:]

                    fout.write(line)
                    atom_count += 1

        expected_atoms = len(self.ref_sequence) * 4
        assert atom_count == expected_atoms, (
            f"Expected {expected_atoms} backbone atoms ({len(self.ref_sequence)} residues), but wrote {atom_count} atoms."
        )
        assert outfile.exists(), f"Backbone PDB not created: {outfile}" # type: ignore
        return outfile


    def run_cg2all(self, folder: Path):
        """
        Runs the cg2all script on each `_backbone.pdb` file in a folder to convert them to all-atom representations.

        Args:
            folder (Path): Path to the folder containing `_backbone.pdb` files.
        """
        folder = Path(folder).resolve()
        backbone_files = list(folder.glob("*_backbone.pdb"))


        for in_pdb in tqdm(backbone_files, desc="Running CG2ALL"): # type: ignore
            out_pdb = in_pdb.with_name(in_pdb.name.replace("_backbone.pdb", "_allatom.pdb"))

            self.cg2all_command = f'''
            eval "$(conda shell.bash hook)"
            conda activate {self.cg2all_environment}
            convert_cg2all -p "{in_pdb}" -o "{out_pdb}" --cg "MainchainModel" --fix --device "{self.cg2all_device}"
            '''

            result = subprocess.run(
                self.cg2all_command,
                shell=True,
                executable="/bin/bash",
                check=False,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"cg2all failed:\n{result.stderr}. Continuing with other files...")
            else:
                in_pdb.unlink()


    def minimize_pdb(self, pdb_file: Path, output_file: Path):
        """Minimize a single PDB file and save the result."""
        pdb = PDBFile(str(pdb_file))
        system = self.forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=NoCutoff,
            constraints=None
        )

        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond) # type: ignore
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        # Energy minimization
        simulation.minimizeEnergy()

        # Save minimized structure
        positions = simulation.context.getState(getPositions=True).getPositions()
        with output_file.open("w") as f:
            PDBFile.writeFile(simulation.topology, positions, f)


    def run_minimization(self, input_dir: Path, output_dir: Path):
        """
        Minimize all PDB structures in input_dir and save them in output_dir
        with '_min.pdb' suffix. If a file fails, print the error and continue.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdb_files = list(input_dir.glob("*_allatom.pdb"))

        for pdb_file in tqdm(pdb_files, desc="Minimizing structures"): # type: ignore
            out_file = output_dir / f"{pdb_file.stem}_min.pdb"
            try:
                self.minimize_pdb(pdb_file, out_file)
            except Exception as e:
                print(f"PDB {pdb_file.name} could not be minimized: {e}")

        print("All minimizations done.")
        
