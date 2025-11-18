import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from datetime import datetime
from uuid import uuid4

class EvaluationStorage:
    """Store and retrieve daily student evaluations"""
    
    def __init__(self, db_path: str = "database/student_evaluations.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _init_db(self):
        """Initialize database with evaluations table"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_evaluations (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                total_submissions INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0.0,
                on_time_rate REAL DEFAULT 0.0,
                participation_score REAL DEFAULT 0.0,
                competence_score REAL DEFAULT 0.0,
                discipline_score REAL DEFAULT 0.0,
                total_score REAL DEFAULT 0.0,
                rating TEXT DEFAULT '',
                teacher_comment TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(student_id, date)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_student_date 
            ON daily_evaluations(student_id, date DESC)
        """)
        
        conn.commit()
        conn.close()
        
        print("✅ Evaluation storage initialized")
    

    def save_evaluation(self, evaluation_data: Dict) -> str:
        """
        Save or update daily evaluation using UUID for new records
        
        Args:
            evaluation_data: Dict with all evaluation fields
            
        Returns:
            evaluation_id (UUID string)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        student_id = evaluation_data.get("student_id")
        date = evaluation_data.get("date")
        
        # Check if evaluation exists
        cursor.execute("""
            SELECT id FROM daily_evaluations
            WHERE student_id = ? AND date = ?
        """, (student_id, date))
        
        existing = cursor.fetchone()
        
        now = datetime.now().isoformat()
        
        if existing:
            # Update existing
            eval_id = existing[0]
            
            cursor.execute("""
                UPDATE daily_evaluations
                SET total_submissions = ?,
                    avg_score = ?,
                    on_time_rate = ?,
                    participation_score = ?,
                    competence_score = ?,
                    discipline_score = ?,
                    total_score = ?,
                    rating = ?,
                    teacher_comment = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                evaluation_data.get("total_submissions", 0),
                evaluation_data.get("avg_score", 0.0),
                evaluation_data.get("on_time_rate", 0.0),
                evaluation_data.get("participation_score", 0.0),
                evaluation_data.get("competence_score", 0.0),
                evaluation_data.get("discipline_score", 0.0),
                evaluation_data.get("total_score", 0.0),
                evaluation_data.get("rating", ""),
                evaluation_data.get("teacher_comment", ""),
                now,
                eval_id
            ))
            
            print(f"   📝 Updated evaluation: {eval_id}")
        else:
            # Insert new with UUID
            unique_part = uuid4().hex[:8]  # 8 characters from UUID
            eval_id = f"eval_{unique_part}"
            
            cursor.execute("""
                INSERT INTO daily_evaluations (
                    id, student_id, date,
                    total_submissions, avg_score, on_time_rate,
                    participation_score, competence_score, discipline_score,
                    total_score, rating, teacher_comment,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval_id,
                student_id,
                date,
                evaluation_data.get("total_submissions", 0),
                evaluation_data.get("avg_score", 0.0),
                evaluation_data.get("on_time_rate", 0.0),
                evaluation_data.get("participation_score", 0.0),
                evaluation_data.get("competence_score", 0.0),
                evaluation_data.get("discipline_score", 0.0),
                evaluation_data.get("total_score", 0.0),
                evaluation_data.get("rating", ""),
                evaluation_data.get("teacher_comment", ""),
                now,
                now
            ))
            
            print(f"   ✅ Saved new evaluation: {eval_id}")
        
        conn.commit()
        conn.close()
        
        return eval_id
    
    def get_evaluation(self, student_id: str, date: str) -> Optional[Dict]:
        """Get evaluation for specific date"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM daily_evaluations
            WHERE student_id = ? AND date = ?
        """, (student_id, date))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_history(
        self,
        student_id: str,
        days: int = 7,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get evaluation history
        
        Args:
            student_id: Student ID
            days: Number of recent days (default: 7)
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of evaluations ordered by date DESC
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if start_date and end_date:
            # Use date range
            cursor.execute("""
                SELECT * FROM daily_evaluations
                WHERE student_id = ?
                AND date >= ? AND date <= ?
                ORDER BY date DESC
            """, (student_id, start_date, end_date))
        else:
            # Use days limit
            cursor.execute("""
                SELECT * FROM daily_evaluations
                WHERE student_id = ?
                ORDER BY date DESC
                LIMIT ?
            """, (student_id, days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_all_students_latest(self) -> List[Dict]:
        """Get latest evaluation for all students"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM daily_evaluations
            WHERE (student_id, date) IN (
                SELECT student_id, MAX(date)
                FROM daily_evaluations
                GROUP BY student_id
            )
            ORDER BY total_score DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]