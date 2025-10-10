"""Session management object.

Provides functionality to record predicted structures and sequences, with custom metadata.
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Union


class SessionTracker:
    """
    Tracks data generated during a session in a hierarchical manner.
    
    Features:
    - Supports nested structure: tracker["run1"]["modelA"]["metrics"]
    - Can record arbitrary dicts
    - Can export all data to JSON
    """
    def __init__(self):
        self._data = defaultdict(dict)

    # ----------------------------
    # 1. RECORDING METHODS
    # ----------------------------

    def record(self, *levels: str, data: Dict[str, Any]):
        """
        Record a dictionary of data under a nested hierarchy.

        Example:
            tracker.record("experiment1", "run2", data={"loss": 0.2, "acc": 0.9})
        """
        # Navigate into nested dicts dynamically
        node = self._data
        for level in levels[:-1]:
            node = node.setdefault(level, {})
        # Merge or create final level
        final_level = levels[-1]
        if final_level not in node:
            node[final_level] = {}
        node[final_level].update(data)

    def update(self, hierarchy: Union[str, list[str]], key: str, value: Any):
        """
        Set a single key/value at a given hierarchy.
        """
        if isinstance(hierarchy, str):
            hierarchy = [hierarchy]
        node = self._data
        for level in hierarchy:
            node = node.setdefault(level, {})
        node[key] = value

    # ----------------------------
    # 2. ACCESS METHODS
    # ----------------------------

    def get(self, *levels: str) -> Dict[str, Any]:
        """
        Retrieve a dictionary stored at a given hierarchy.
        """
        node = self._data
        for level in levels:
            node = node[level]
        return node

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the entire session data as a regular dictionary.
        """
        return json.loads(json.dumps(self._data))  # convert defaultdict → dict

    # ----------------------------
    # 3. SAVE / LOAD
    # ----------------------------

    def save_json(self, filepath: Union[str, Path], indent: int = 2):
        """
        Write the tracked data to a JSON file.
        """
        path = Path(filepath)
        with path.open("w") as f:
            json.dump(self.as_dict(), f, indent=indent)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> "SessionTracker":
        """
        Load a tracker from a JSON file.
        """
        path = Path(filepath)
        with path.open("r") as f:
            data = json.load(f)
        tracker = cls()
        tracker._data.update(data)
        return tracker

    # ----------------------------
    # 4. DEBUG / PRETTY PRINT
    # ----------------------------

    def __repr__(self):
        return json.dumps(self.as_dict(), indent=2)



