from .proteinmpnn import ProteinMPNNPredictor


class SequencePredictorRegistry:
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


SequencePredictorRegistry.register("proteinmpnn", ProteinMPNNPredictor)