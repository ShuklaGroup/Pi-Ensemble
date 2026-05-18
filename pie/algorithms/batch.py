from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from Levenshtein import distance as edit_distance
import tqdm
import subprocess
from openmm.app import ForceField, PDBFile, Simulation, NoCutoff
from openmm import LangevinIntegrator
from openmm.unit import kelvin, picosecond # type: ignore

from .base import DEFAULT_MINIMIZE_FORCEFIELD, InterpolationAlgorithm, normalize_forcefield_files
from ..alignment_utils import compute_alignment_indices
from ..constants import ONE_TO_THREE
from ..io_utils import read_alignment_indices
from ..sequence.base import SequencePredictor
from ..session.session import SessionTracker
from ..structure.base import StructurePredictor


TemplateRecord = Dict[str, object]


class BatchInterpolation(InterpolationAlgorithm):
    def __init__(
        self,
        ref_sequence: str,
        number_steps: int,
        structure_model: StructurePredictor,
        sequence_model: SequencePredictor,
        template_1: Path,
        chain_id_1: str,
        template_2: Path,
        chain_id_2: str,
        outpath: Path,
        min_edit: int = 1,
        cg2all: bool = False,
        minimize: bool = False,
        minimize_forcefield: Union[str, Sequence[str]] = DEFAULT_MINIMIZE_FORCEFIELD,
        **kwargs,
    ):
        self.ref_sequence = ref_sequence
        self.number_steps = number_steps
        self.structure_model = structure_model
        self.sequence_model = sequence_model
        self.outpath = Path(outpath)
        self.min_edit = min_edit
        self.cg2all = cg2all
        self.minimize = minimize
        self.minimize_forcefield = normalize_forcefield_files(minimize_forcefield)

        if self.cg2all:
            self.cg2all_environment = kwargs.get("cg2all_environment", "cg2all")
            # self.cg2all_device = kwargs.get("cg2all_device", "cpu")
            self.cg2all_device = "cpu" # We have only implemented CPU support for cg2all for now, since GPU support requires additional setup and testing.
            if self.minimize:
                self.forcefield = ForceField(*self.minimize_forcefield)

        self.template_1 = self._process_template(Path(template_1), chain_id_1, "template_1")
        self.template_2 = self._process_template(Path(template_2), chain_id_2, "template_2")

        aln_file = kwargs.get("aln_file")
        if aln_file is not None:
            self.alignment_map = read_alignment_indices(aln_file, self.ref_sequence)
        else:
            seqs = [
                str(self.template_1["modeled_seq"]),
                str(self.template_2["modeled_seq"]),
            ]
            self.alignment_map = compute_alignment_indices(seqs, self.ref_sequence)

        self._align_prob_dist(self.template_1, self.template_2)

        self.logger = SessionTracker()
        self.logger.record("templates", "template_1", data=self._serializable_record(self.template_1))
        self.logger.record("templates", "template_2", data=self._serializable_record(self.template_2))

        self.rounds: List[List[TemplateRecord]] = [[self.template_1, self.template_2]]

    def _process_template(self, template: Path, chain_id: str, name: str) -> TemplateRecord:
        template_dict: TemplateRecord = {
            "name": name,
            "struct_path": template,
            "chain_id": chain_id,
        }

        outpath = self.outpath / "round_0" / name
        outpath.mkdir(parents=True, exist_ok=True)
        seq_pred = self.predict_sequence({"struct_path": template}, outpath, chain_id=chain_id)
        template_dict.update(seq_pred)
        return template_dict

    def _align_prob_dist(self, template_1: TemplateRecord, template_2: TemplateRecord) -> None:
        aligned_seqs = self.alignment_map["aligned_seqs"]
        index_maps = self.alignment_map["index_maps"]

        template_1["aligned_seq"] = aligned_seqs[0]
        template_2["aligned_seq"] = aligned_seqs[1]
        template_1["index_map"] = index_maps[0]
        template_2["index_map"] = index_maps[1]

        total_length = len(aligned_seqs[0])
        alphabet_size = len(self.sequence_model.alphabet) # type: ignore

        for template in (template_1, template_2):
            new_dist = np.zeros((total_length, alphabet_size))
            index_map = template["index_map"]
            prob_dist = template["prob_dist"]
            assert isinstance(index_map, list)
            assert isinstance(prob_dist, np.ndarray)
            for i, mapped in enumerate(index_map):
                if mapped is not None:
                    new_dist[i] = prob_dist[mapped]
            template["prob_dist"] = new_dist

        shape_1 = template_1["prob_dist"]
        shape_2 = template_2["prob_dist"]
        assert isinstance(shape_1, np.ndarray)
        assert isinstance(shape_2, np.ndarray)
        if shape_1.shape != shape_2.shape:
            raise ValueError("Aligned probability distributions must have matching shapes.")

    def predict_sequence(self, structure: dict, outpath: Union[str, Path], **kwargs):
        return self.sequence_model.predict(structure["struct_path"], outpath, **kwargs) # type: ignore

    def predict_structure(self, sequence: str, outpath: Union[str, Path], **kwargs):
        return self.structure_model.predict(sequence, outpath, **kwargs) # type: ignore

    def predict_structures(
        self, sequences: List[str], outpaths: List[Union[str, Path]], **kwargs
    ) -> List[dict]:
        predict_batch = getattr(self.structure_model, "predict_batch", None)
        if callable(predict_batch):
            return predict_batch(sequences, outpaths, **kwargs) # type: ignore

        return [
            self.predict_structure(sequence, outpath, **kwargs)
            for sequence, outpath in zip(sequences, outpaths)
        ]

    def mix_probabilities(self, seq_1: dict, seq_2: dict, weight: float, **kwargs):
        prob_1 = seq_1["prob_dist"]
        prob_2 = seq_2["prob_dist"]
        assert isinstance(prob_1, np.ndarray)
        assert isinstance(prob_2, np.ndarray)
        prob_dist = weight * prob_1 + (1.0 - weight) * prob_2
        return self._decode_sequence(prob_dist)

    def find_crit_lambdas(self, template_1: TemplateRecord, template_2: TemplateRecord) -> np.ndarray:
        prob_1 = template_1["prob_dist"]
        prob_2 = template_2["prob_dist"]
        assert isinstance(prob_1, np.ndarray)
        assert isinstance(prob_2, np.ndarray)

        numerator = prob_2[:, :, None] - prob_2[:, None, :]
        diff = prob_1 - prob_2
        denominator = diff[:, :, None] - diff[:, None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            lambdas = numerator / denominator

        finite = np.isfinite(lambdas)
        in_range = (lambdas >= 0.0) & (lambdas <= 1.0)
        lambda_crit = np.sort(lambdas[finite & in_range])

        if lambda_crit.size == 0:
            return np.asarray([0.5], dtype=float)

        previous = np.concatenate(([0.0], lambda_crit[:-1]))
        lambda_midpoints = (previous + lambda_crit) / 2.0
        lambda_midpoints = np.unique(lambda_midpoints)

        if lambda_midpoints[0] > 0.0:
            lambda_midpoints = np.concatenate(([0.0], lambda_midpoints))
        if lambda_midpoints[-1] < 1.0:
            lambda_midpoints = np.concatenate((lambda_midpoints, [1.0]))
        return lambda_midpoints

    def compute_edit_distance(
        self, seq_dict: Dict[str, float], min_edit: int
    ) -> Dict[str, Tuple[float, int]]:
        filtered: Dict[str, Tuple[float, int]] = {}
        prev_seq = None

        for seq, val in seq_dict.items():
            if prev_seq is None:
                filtered[seq] = (val, 0)
                prev_seq = seq
            else:
                dist = edit_distance(prev_seq, seq)
                if dist >= min_edit:
                    filtered[seq] = (val, dist)
                    prev_seq = seq

        return filtered

    def find_interpolated_sequences(
        self,
        lambda_crit_inter: np.ndarray,
        template_1: TemplateRecord,
        template_2: TemplateRecord,
        min_edit: int | None = None,
    ) -> Dict[str, Tuple[float, int]]:
        prob_1 = template_1["prob_dist"]
        prob_2 = template_2["prob_dist"]
        assert isinstance(prob_1, np.ndarray)
        assert isinstance(prob_2, np.ndarray)

        min_edit = self.min_edit if min_edit is None else min_edit
        probs_all = (
            lambda_crit_inter[None, None, :] * prob_1[:, :, None]
            + (1.0 - lambda_crit_inter[None, None, :]) * prob_2[:, :, None]
        ).transpose(2, 0, 1)

        seqs: Dict[str, float] = {}
        for probs, lam in zip(probs_all, lambda_crit_inter):
            seq = self._decode_sequence(probs)
            seqs[seq] = float(lam)

        return self.compute_edit_distance(seqs, min_edit)

    def find_anchors(self, num_round: int) -> List[Tuple[TemplateRecord, TemplateRecord]]:
        round_zero = self.rounds[0]
        template_1 = round_zero[0]
        template_2 = round_zero[1]

        if num_round == 1:
            return [(template_1, template_2)]

        previous_round = self.rounds[num_round - 1]
        anchors: List[Tuple[TemplateRecord, TemplateRecord]] = []

        if num_round == 2:
            common_anchor = self._select_max_edit_record(previous_round)
            return [(template_1, common_anchor), (template_2, common_anchor)]

        if previous_round:
            previous_a = [record for record in previous_round if record.get("direction") == "A"]
            previous_b = [record for record in previous_round if record.get("direction") == "B"]

            if previous_a:
                anchors.append((template_1, self._select_max_edit_record(previous_a)))
            if previous_b:
                anchors.append((template_2, self._select_max_edit_record(previous_b)))

        if not anchors:
            anchors.append((template_1, template_2))
        return anchors

    def run(self):
        generated_structs: List[TemplateRecord] = []

        for round_idx in range(1, self.number_steps + 1):
            anchor_pairs = self.find_anchors(round_idx)
            generated_round: List[TemplateRecord] = []

            for pair_idx, (anchor, mobile) in enumerate(anchor_pairs):
                direction = "A" if anchor["struct_path"] == self.template_1["struct_path"] else "B"
                lambda_values = self.find_crit_lambdas(anchor, mobile)
                interpolated = self.find_interpolated_sequences(lambda_values, anchor, mobile)

                sequences = list(interpolated.items())
                generated_structs: List[TemplateRecord] = []
                edit_distances: List[int] = []

                sequence_strings = [sequence for sequence, _ in sequences]
                struct_outpaths = [
                    self.outpath
                    / f"round_{round_idx}"
                    / f"direction_{direction}"
                    / f"sequence_{seq_idx:03d}"
                    / "structure.pdb"
                    for seq_idx, _ in enumerate(sequences)
                ]
                for struct_outpath in struct_outpaths:
                    struct_outpath.parent.mkdir(parents=True, exist_ok=True)

                predicted_structures = self.predict_structures(sequence_strings, struct_outpaths) # type: ignore

                for seq_idx, ((sequence, (lam, dist)), new_struct, struct_path) in enumerate(
                    zip(sequences, predicted_structures, struct_outpaths)
                ):
                    seq_outpath = struct_path.parent
                    seq_pred = self.predict_sequence(new_struct, seq_outpath)

                    record: TemplateRecord = {
                        **new_struct,
                        **seq_pred,
                        "direction": direction,
                        "round": round_idx,
                        "pair_index": pair_idx,
                        "sequence_index": seq_idx,
                        "lambda": float(lam),
                        "edit_distance": int(dist),
                        "source_anchor": str(anchor["struct_path"]),
                        "source_mobile": str(mobile["struct_path"]),
                    }
                    generated_structs.append(record)
                    edit_distances.append(int(dist))
                    generated_round.append(record)

                    self.logger.record(
                        f"round_{round_idx}",
                        f"direction_{direction}",
                        f"sequence_{seq_idx:03d}",
                        data=self._serializable_record(record),
                    )

                self.logger.record(
                    f"round_{round_idx}",
                    f"direction_{direction}",
                    "summary",
                    data={
                        "num_sequences": len(sequences),
                        "lambda_values": [float(item[1][0]) for item in sequences],
                        "edit_distances": edit_distances,
                    },
                )

            self.rounds.append(generated_round)
            self.logger.save_json(self.outpath / "log.json")

        if self.cg2all:
            cg2all_outpath = self.outpath / "cg2all"
            for struct in generated_structs:
                self.extract_backbone_coords_to_pdb(struct, cg2all_outpath)
            self.run_cg2all(cg2all_outpath)

            if self.minimize:
                min_outpath = self.outpath / "minimized"
                self.run_minimization(cg2all_outpath, min_outpath)

    def _select_max_edit_record(self, records: List[TemplateRecord]) -> TemplateRecord:
        if not records:
            raise ValueError("Cannot select an anchor from an empty round.")
        return max(records, key=lambda record: int(record.get("edit_distance", 0))) # type: ignore

    def _decode_sequence(self, prob_dist: np.ndarray) -> str:
        alphabet = np.asarray(self.sequence_model.alphabet)
        argmax = np.argmax(prob_dist, axis=1)
        return "".join(alphabet[argmax])

    def _serializable_record(self, record: TemplateRecord) -> Dict[str, object]:
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
        keep = {"N", "CA", "C", "O"}
        in_path = Path(structure["struct_path"])
        round_idx = structure.get("round", "unknown")
        direction = structure.get("direction", "X")
        sequence_idx = structure.get("sequence_index", "unknown")
        outfile = outpath / f"round_{round_idx}_direction_{direction}_sequence_{sequence_idx}_{in_path.stem}_backbone.pdb"
        outfile.parent.mkdir(parents=True, exist_ok=True)

        residue_index = -1
        atom_count = 0

        with open(in_path, "r") as fin, open(outfile, "w") as fout:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() in keep:
                    atom_name = line[12:16].strip()

                    if atom_name == "N":
                        residue_index += 1
                        if residue_index >= len(self.ref_sequence):
                            raise ValueError(
                                f"Too many residues in PDB file for given reference sequence of length {len(self.ref_sequence)}."
                            )

                    if residue_index < len(self.ref_sequence):
                        new_resname = ONE_TO_THREE[self.ref_sequence[residue_index]]
                        line = line[:17] + f"{new_resname:>3}" + line[20:]

                    fout.write(line)
                    atom_count += 1

        expected_atoms = len(self.ref_sequence) * 4
        assert (atom_count == expected_atoms or atom_count == expected_atoms - 1), (
            f"Expected {expected_atoms} backbone atoms ({len(self.ref_sequence)} residues), but wrote {atom_count} atoms."
        )
        assert outfile.exists(), f"Backbone PDB not created: {outfile}" # type: ignore
        return outfile

    def run_cg2all(self, folder: Path):
        folder = Path(folder).resolve()
        backbone_files = list(folder.glob("*_backbone.pdb"))

        for in_pdb in tqdm.tqdm(backbone_files, desc="Running CG2ALL"): # type: ignore
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
        pdb = PDBFile(str(pdb_file))
        system = self.forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=NoCutoff,
            constraints=None
        )

        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond) # type: ignore
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()

        positions = simulation.context.getState(getPositions=True).getPositions()
        with output_file.open("w") as f:
            PDBFile.writeFile(simulation.topology, positions, f)

    def run_minimization(self, input_dir: Path, output_dir: Path):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdb_files = list(input_dir.glob("*_allatom.pdb"))

        for pdb_file in tqdm.tqdm(pdb_files, desc="Minimizing structures"): # type: ignore
            out_file = output_dir / f"{pdb_file.stem}_min.pdb"
            try:
                self.minimize_pdb(pdb_file, out_file)
            except Exception as e:
                print(f"PDB {pdb_file.name} could not be minimized: {e}")

        print("All minimizations done.")
