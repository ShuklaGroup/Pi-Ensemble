import argparse
import yaml
from pathlib import Path
from pie.io_utils import read_fasta_first_seq
from pie.algorithms.registry import AlgorithmRegistry
from pie.structure.registry import StructurePredictorRegistry
from pie.sequence.registry import SequencePredictorRegistry


def getargs():
    parser = argparse.ArgumentParser(description="Interpolate between structural templates.")
    parser.add_argument("config", type=str, help="Path to YAML configuration file (see docs).")
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


    # Required arguments
    # parser.add_argument(
    #     "ref_seq",
    #     type=Path,
    #     help="Path to reference sequence in FASTA format. The first sequence will be used."
    # )
    # parser.add_argument(
    #     "template_1",
    #     type=Path,
    #     help="Path to first structural template (PDB) for interpolation."
    # )
    # parser.add_argument(
    #     "template_2",
    #     type=Path,
    #     help="Path to second structural template (PDB) for interpolation."
    # )

    # Optional arguments with defaults
    # parser.add_argument(
    #     "--chain_id_1",
    #     type=str,
    #     help="Chain ID from first template file.",
    #     default="A"
    # )
    # parser.add_argument(
    #     "--chain_id_2",
    #     type=str,
    #     help="Chain ID from second template file.",
    #     default="A"
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=Path,
    #     default=Path("output"),
    #     help="Output directory (default: output)."
    # )
    # parser.add_argument(
    #     "--rounds",
    #     type=int,
    #     default=10,
    #     help="Number of interpolation rounds (default: 10)."
    # )
    # parser.add_argument(
    #     "--interpolation_steps",
    #     type=int,
    #     default=100,
    #     help="Number of interpolation steps (default: 100)."
    # )
    # parser.add_argument(
    #     "--structure_model",
    #     choices=["esm3", "boltz2"],
    #     help="Choice of structure prediction model."
    # )
    # parser.add_argument(
    #     "--ligands",
    #     type=Path,
    #     default=None,
    #     help="Path to OPTIONAL ligand FASTA file (CCD or SMILES format). Default: None."
    # )
    # parser.add_argument(
    #     "--pmpnn_path",
    #     type=Path,
    #     default=Path("/opt/ProteinMPNN"),
    #     help="Path to protein_mpnn_run.py (default: /opt/ProteinMPNN)."
    # )
    
    # parser.add_argument(
    #     "--min_edit_dist",
    #     type=int,
    #     default=1,
    #     help="Minimum edit distance between sequences considered in a single round (default: 1 = all sequences). Used only with --batch_msa."
    # )

    # parser.add_argument(
    #     "--temperature",
    #     type=float,
    #     default=0.7,
    #     help="Temperature for ESM3 (default: 0.7). Used only with --esm3."
    # )
    # parser.add_argument(
    #     "--boltz_script",
    #     type=Path,
    #     default=Path("./boltz.sh"),
    #     help="Path to boltz-2 executable script (default: ./boltz.sh)."
    # )
    # parser.add_argument(
    #     "--pmpnn_script",
    #     type=Path,
    #     default=Path("./pmpnn.sh"),
    #     help="Path to protein mpnn executable script (default: ./pmpnn.sh)."
    # )
    # parser.add_argument(
    #     "--cg2all_script",
    #     type=Path,
    #     default=Path("./cg2all.sh"),
    #     help="Path to protein cg2all executable script (default: ./cg2all.sh)."
    # )
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     choices=["cpu", "gpu"],
    #     default="gpu",
    #     help="Device to use for computation: 'cpu' or 'gpu' (default: gpu)."
    # )
    # parser.add_argument(
    #     "--template_alignment",
    #     type=Path,
    #     default=None,
    #     help="Path to custom alignment for template structures (.fa/.fasta)."
    # )
    # parser.add_argument(
    #     "--msa_mode",
    #     type=str,
    #     choices=["server", "local", "empty"],
    #     default="server",
    #     help="MSA computation mode (Boltz only). Local is not implemented yet."
    # )
    # parser.add_argument(
    #     "--esm3",
    #     action="store_true",
    #     help="Run using ESM3 instead of Boltz (no MSA required, faster). (default: False)"
    # )
    # parser.add_argument(
    #     "--batch_msa",
    #     action="store_true",
    #     help="Run using batch MSA query (Boltz only). (default: False)"
    # )
    # parser.add_argument(
    #     "--cg2all",
    #     action="store_true",
    #     help="Run CG2ALL postprocessing to pack reference sequence side chains in generated backbones. (default: False)"
    # )
    # parser.add_argument(
    #     "--minimize",
    #     action="store_true",
    #     help="Run OpenMM to minimize structure energy. Requires --cg2all. (default: False)"
    # )

    # return parser.parse_args()



def main():
    cfg = getargs()

    # Init models
    struct_cfg = cfg["structure_prediction"]
    structure_model = StructurePredictorRegistry.create(struct_cfg["model"], **struct_cfg.get("kwargs", {}))

    seq_cfg = cfg["sequence_prediction"]
    sequence_model = SequencePredictorRegistry.create(seq_cfg["model"], **seq_cfg.get("kwargs", {}))

    # Init algorithm
    algo_cfg = cfg["interpolation"]
    algo_kwargs = {
        **algo_cfg.get("kwargs", {}), 
        "structure_model": structure_model, 
        "sequence_model": sequence_model
    }
    algorithm = AlgorithmRegistry.create(algo_cfg["name"], **algo_kwargs)

    # Run!
    algorithm.run()