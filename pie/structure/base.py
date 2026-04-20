from abc import ABC, abstractmethod
from pathlib import Path



class StructurePredictor(ABC):
    """Base class for structure prediction models."""

    # All concrete classes must return at least these elements with self._predict()
    REQUIRED_KEYS = {
        "struct_path": Path,
        "sequence": str,
    }

    def predict(self, sequence: str, **kwargs) -> dict:
        result = self._predict(sequence, **kwargs)
        self._validate_dict(result)
        return result


    def _validate_dict(self, result: dict):
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict, got {type(result)}")
        for key, expected_type in self.REQUIRED_KEYS.items():
            if key not in result:
                raise KeyError(f"Missing required key: '{key}'")
            if not isinstance(result[key], expected_type):
                raise TypeError(
                    f"Key '{key}' must be {expected_type}, got {type(result[key])}"
                )
                

    @abstractmethod
    def _predict(self, sequence: str, **kwargs) -> dict:
        """
        Predict the protein structure from a sequence.
        
        Parameters:
            sequence (str): Protein sequence (single-letter amino acids).
            **kwargs: Model-specific parameters.
        
        Returns:
            object: A model-specific structure representation.
        """
        pass