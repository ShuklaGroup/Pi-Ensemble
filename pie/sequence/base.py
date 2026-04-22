from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import numpy as np



class SequencePredictor(ABC):
    """Base class for sequence prediction models."""

    # All concrete classes must return at least these elements with self._predict()
    REQUIRED_KEYS = {
        "prob_dist": np.ndarray,
        "sequence": str,
    }

    @property
    @abstractmethod
    def alphabet(self):
        """Return the amino-acid alphabet used by this predictor."""
        pass

    def predict(self, structure: Union[str, Path], outpath: Union[str, Path], **kwargs) -> dict:
        result = self._predict(structure, outpath, **kwargs)
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
    def _predict(self, structure: Union[str, Path], outpath: Union[str, Path], **kwargs) -> dict:
        """
        Predict the protein sequence from a structure.
        
        Parameters:
            structure (str, Path): Path to protein structure.
            outpath (str, Path): Output directory for model artifacts.
            **kwargs: Model-specific parameters.
        
        Returns:
            object: A model-specific sequence representation.
        """
        pass
