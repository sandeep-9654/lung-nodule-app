"""
CSV Tools - Utilities for reading CSV files.
Used by the LUNA16 FROC evaluation script.
"""

import csv
from typing import List


def readCSV(filename: str) -> List[List[str]]:
    """
    Read a CSV file and return its contents as a list of rows.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        List of rows, where each row is a list of string values
    """
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            lines.append(row)
    return lines


def writeCSV(filename: str, data: List[List], header: List[str] = None):
    """
    Write data to a CSV file.
    
    Args:
        filename: Path to the output CSV file
        data: List of rows to write
        header: Optional header row
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)
