"""
SQLite Database Module
Manages Patients and DiagnosticReports tables for the nodule detection system.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'nodule_detection.db')


def get_db_path() -> str:
    """Get absolute path to database file."""
    db_path = os.path.abspath(DATABASE_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return db_path


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize database with required tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                scan_date DATETIME,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Diagnostic Reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnostic_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER REFERENCES patients(id),
                scan_path TEXT,
                scan_filename TEXT,
                nodule_count INTEGER DEFAULT 0,
                nodule_locations TEXT,
                max_confidence REAL,
                model_version TEXT DEFAULT '1.0',
                inference_time_ms INTEGER,
                status TEXT DEFAULT 'completed',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER,
                train_loss REAL,
                val_loss REAL,
                dice_score REAL,
                sensitivity REAL,
                specificity REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print(f"Database initialized at: {get_db_path()}")


# Patient CRUD operations
def create_patient(name: str, age: int = None, gender: str = None, 
                   scan_date: datetime = None, notes: str = None) -> int:
    """Create a new patient record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patients (name, age, gender, scan_date, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, age, gender, scan_date, notes))
        conn.commit()
        return cursor.lastrowid


def get_patient(patient_id: int) -> Optional[Dict]:
    """Get patient by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_patients() -> List[Dict]:
    """Get all patients."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
        return [dict(row) for row in cursor.fetchall()]


# Diagnostic Report operations
def create_report(
    scan_path: str,
    scan_filename: str,
    nodule_count: int,
    nodule_locations: List[Dict],
    max_confidence: float,
    inference_time_ms: int,
    patient_id: int = None
) -> int:
    """Create a new diagnostic report."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO diagnostic_reports 
            (patient_id, scan_path, scan_filename, nodule_count, 
             nodule_locations, max_confidence, inference_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, scan_path, scan_filename, nodule_count,
            json.dumps(nodule_locations), max_confidence, inference_time_ms
        ))
        conn.commit()
        return cursor.lastrowid


def get_report(report_id: int) -> Optional[Dict]:
    """Get diagnostic report by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM diagnostic_reports WHERE id = ?', (report_id,))
        row = cursor.fetchone()
        if row:
            report = dict(row)
            if report['nodule_locations']:
                report['nodule_locations'] = json.loads(report['nodule_locations'])
            return report
        return None


def get_all_reports(limit: int = 100) -> List[Dict]:
    """Get all diagnostic reports."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM diagnostic_reports 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        reports = []
        for row in cursor.fetchall():
            report = dict(row)
            if report['nodule_locations']:
                report['nodule_locations'] = json.loads(report['nodule_locations'])
            reports.append(report)
        return reports


# Model metrics operations
def save_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float = None,
    dice_score: float = None,
    sensitivity: float = None,
    specificity: float = None
) -> int:
    """Save training metrics."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO model_metrics 
            (epoch, train_loss, val_loss, dice_score, sensitivity, specificity)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (epoch, train_loss, val_loss, dice_score, sensitivity, specificity))
        conn.commit()
        return cursor.lastrowid


def get_training_history() -> List[Dict]:
    """Get all training metrics."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM model_metrics ORDER BY epoch')
        return [dict(row) for row in cursor.fetchall()]


def get_latest_metrics() -> Optional[Dict]:
    """Get the most recent training metrics."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM model_metrics 
            ORDER BY epoch DESC 
            LIMIT 1
        ''')
        row = cursor.fetchone()
        return dict(row) if row else None


def get_statistics() -> Dict[str, Any]:
    """Get overall system statistics."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Count patients
        cursor.execute('SELECT COUNT(*) as count FROM patients')
        patient_count = cursor.fetchone()['count']
        
        # Count reports
        cursor.execute('SELECT COUNT(*) as count FROM diagnostic_reports')
        report_count = cursor.fetchone()['count']
        
        # Average confidence
        cursor.execute('SELECT AVG(max_confidence) as avg FROM diagnostic_reports')
        avg_confidence = cursor.fetchone()['avg'] or 0
        
        # Total nodules detected
        cursor.execute('SELECT SUM(nodule_count) as total FROM diagnostic_reports')
        total_nodules = cursor.fetchone()['total'] or 0
        
        return {
            'patient_count': patient_count,
            'report_count': report_count,
            'avg_confidence': round(avg_confidence, 3),
            'total_nodules_detected': total_nodules
        }


# Initialize on import
if __name__ == "__main__":
    init_database()
    print("\nDatabase module test:")
    print("-" * 50)
    
    # Test creating a patient
    patient_id = create_patient("Test Patient", age=45, gender="M")
    print(f"Created patient with ID: {patient_id}")
    
    # Test creating a report
    report_id = create_report(
        scan_path="/test/scan.mhd",
        scan_filename="scan.mhd",
        nodule_count=2,
        nodule_locations=[
            {"centroid": [32, 128, 128], "probability": 0.95},
            {"centroid": [48, 200, 180], "probability": 0.87}
        ],
        max_confidence=0.95,
        inference_time_ms=12500,
        patient_id=patient_id
    )
    print(f"Created report with ID: {report_id}")
    
    # Get statistics
    stats = get_statistics()
    print(f"Statistics: {stats}")
    
    print("\n✓ Database module test passed!")
