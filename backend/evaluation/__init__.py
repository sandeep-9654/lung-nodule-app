"""
LUNA16 FROC Evaluation Package
Provides tools for computing FROC curves and evaluating CAD systems.
"""

from .NoduleFinding import NoduleFinding
from .csvTools import readCSV, writeCSV
from .froc_evaluation import (
    computeFROC,
    computeFROC_bootstrap,
    evaluateCAD,
    noduleCADEvaluation,
    FROC_minX,
    FROC_maxX
)

__all__ = [
    'NoduleFinding',
    'readCSV',
    'writeCSV',
    'computeFROC',
    'computeFROC_bootstrap',
    'evaluateCAD',
    'noduleCADEvaluation',
    'FROC_minX',
    'FROC_maxX'
]
