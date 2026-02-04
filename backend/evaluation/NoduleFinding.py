"""
NoduleFinding - Data class for nodule annotations and CAD marks.
Used by the LUNA16 FROC evaluation script.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NoduleFinding:
    """Represents a nodule finding (ground truth annotation or CAD mark)."""
    
    # Coordinates
    coordX: float = 0.0
    coordY: float = 0.0
    coordZ: float = 0.0
    
    # Size
    diameter_mm: float = 0.0
    
    # CAD probability (for detector outputs)
    CADprobability: float = 0.0
    
    # Candidate ID (for tracking)
    candidateID: int = -1
    id: int = -1
    
    # State: "Included", "Excluded", or ""
    state: str = ""
    
    def __repr__(self) -> str:
        return (
            f"NoduleFinding(x={self.coordX:.2f}, y={self.coordY:.2f}, z={self.coordZ:.2f}, "
            f"d={self.diameter_mm:.2f}mm, prob={self.CADprobability:.4f}, state={self.state})"
        )
