from .esm3 import ESM3Predictor
from .esmfold2 import ESMFold2Predictor
from .boltz import BoltzPredictor
from .bioemu import BioEmuPredictor


class StructurePredictorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, predictor_cls):
        cls._registry[name.lower()] = predictor_cls

    @classmethod
    def create(cls, name: str, **kwargs):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown predictor: {name}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls):
        return list(cls._registry.keys())


StructurePredictorRegistry.register("esm3", ESM3Predictor)
StructurePredictorRegistry.register("esmfold2", ESMFold2Predictor)
StructurePredictorRegistry.register("boltz", BoltzPredictor)
StructurePredictorRegistry.register("bioemu", BioEmuPredictor)
