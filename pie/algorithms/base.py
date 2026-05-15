from abc import ABC, abstractmethod
from typing import Sequence, Union


DEFAULT_MINIMIZE_FORCEFIELD = "charmm36_2024.xml"


def normalize_forcefield_files(forcefield: Union[str, Sequence[str]]) -> tuple[str, ...]:
	if isinstance(forcefield, str):
		return (forcefield,)

	forcefield_files = tuple(forcefield)
	if not forcefield_files:
		raise ValueError("minimize_forcefield must include at least one OpenMM forcefield XML file.")
	return forcefield_files



class InterpolationAlgorithm(ABC):


	@abstractmethod
	def predict_structure(**kwargs):
		pass


	@abstractmethod
	def predict_sequence(**kwargs):
		pass


	@abstractmethod
	def mix_probabilities(**kwargs):
		pass


	@abstractmethod
	def run(**kwargs):
		pass
