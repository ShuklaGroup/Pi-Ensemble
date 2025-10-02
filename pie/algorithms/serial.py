from .base import InterpolationAlgorithm
from ..structure.base import StructurePredictor
from ..sequence.base import SequencePredictor
from typing import List
from pathlib import Path



class SerialInterpolation(InterpolationAlgorithm):

	def __init__(
		number_steps: int, 
		mixing_weights: List[float], 
		structure_model: StructurePredictor, 
		sequence_model: SequencePredictor,
		template_1: Path,
		template_2: Path,
		**kwargs
	):