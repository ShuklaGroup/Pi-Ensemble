from abc import ABC, abstractmethod



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
	def run_loop(**kwargs):
		pass
