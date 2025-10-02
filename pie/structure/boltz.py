from .base import StructurePredictor



class BoltzPredictor(StructurePredictor):


	def _predict(self, sequence: str, **kwargs):
		raise NotImplementedError