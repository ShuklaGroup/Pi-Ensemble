import numpy as np
from Bio.Align import PairwiseAligner
from Bio import AlignIO
from typing import List, Union
from pathlib import Path
from .base import InterpolationAlgorithm
from ..session import SessionTracker
from ..structure.base import StructurePredictor
from ..sequence.base import SequencePredictor
from ..io_utils import read_alignment_indices
from ..alignment_utils import compute_alignment_indices



class SerialInterpolation(InterpolationAlgorithm):

    def __init__(
        self,
        ref_sequence: str,
        number_steps: int, 
        mixing_weights: List[float], 
        structure_model: StructurePredictor, 
        sequence_model: SequencePredictor,
        template_1: Path,
        chain_id_1: str,
        template_2: Path,
        chain_id_2: str,
        outpath: Path,
        cg2all: Bool = False,
        minimize: Bool = False,
        **kwargs
    ):

    self.ref_sequence = ref_sequence
    self.number_steps = number_steps
    self.mixing_weights = mixing_weights
    self.structure_model = structure_model
    self.sequence_model = sequence_model
    self.outpath = outpath
    self.cg2all = cg2all
    self.minimize = minimize
    
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

    self.logger.record("templates", "template_1", self.template_1)
    self.logger.record("templates", "template_2", self.template_1)


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
        for temp in (self.template_1, self.template_2):
            new_dist = np.zeros((total_length, 21))
            for i in range(total_length):
                mapped = temp['index_map'][i]
                if mapped is not None:
                    new_dist[i] = temp['prob_dist'][mapped]
            temp['prob_dist'] = new_dist

        assert(self.template_1['prob_dist'].shape == self.template_2['prob_dist'].shape)


    def predict_sequence(self, structure: dict, outpath: Union[str, Path], **kwargs):
        return self.sequence_model.predict(structure['struct_path'], outpath, **kwargs)


    def predict_structure(self, sequence: str, outpath: Union[str, Path], **kwargs):
        return self.structure_model.predict(sequence, outpath, **kwargs)


    def mix_probabilities(self, seq_1: dict, seq_2: dict, weight: float, **kwargs):
        
        prob_dist = weight*seq_1['prob_dist'] + (1-weight)*seq_2['prob_dist']

        sequence = ''.join(
            self.sequence_model.alphabet[np.argmax(p)] 
            for p in prob_dist
        )

        return sequence


    def run(self):

        for weight in self.mixing_weights:
            outpath = self.outpath / f"weight_{repr(weight)}"
            for direction in ['A', 'B']:
                outpath /= f"direction_{direction}"

                if direction == 'A':
                    anchor = self.template_1
                    mobile = self.template_2
                else:
                    anchor = self.template_2
                    mobile = self.template_1

                for step in range(1, self.number_steps + 1):
                    outpath /= f"round_{repr(step)}"

                    # Mix probs
                    new_seq = self.mix_probabilities(anchor, mobile, weight)

                    # Predict structure
                    struct_path = outpath / "structure.pdb"
                    new_struct = self.predict_structure(new_seq, struct_path)
                    self.logger.record(f"weight_{repr(weight)}", f"direction_{direction}", f"round_{repr(step)}", "struct_pred", new_struct)

                    # Predict sequence
                    mobile = self.predict_sequence(new_struct, outpath)
                    self.logger.record(f"weight_{repr(weight)}", f"direction_{direction}", f"round_{repr(step)}", "seq_pred", mobile)

                self.logger.save_json(self.outpath / "log.json")


    def run_cg2all():
        raise NotImplementedError()



    def run_minimization():
        raise NotImplementedError()