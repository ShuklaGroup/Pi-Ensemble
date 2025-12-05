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