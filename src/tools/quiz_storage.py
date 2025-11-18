"""
src/tools/quiz_storage.py

SQLite-based storage for quiz history
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json


class QuizStorage:
    """Manage quiz storage in SQLite database"""
    
    def __init__(self, db_path: str = "database/quiz_storage.db"):  # ← ĐỔI PATH
        self.db_path = Path(db_path)
        self._ensure_database()
    
    def _ensure_database(self):
        """Create database and tables if not exist"""
        # Create database directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create quizzes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                daily_count INTEGER NOT NULL,
                content TEXT NOT NULL,
                subject TEXT,
                topic TEXT,
                difficulty TEXT,
                num_questions INTEGER DEFAULT 10,
                time_limit INTEGER DEFAULT 15,
                answer_key TEXT,           -- ← NEW
                status TEXT DEFAULT 'pending',  -- ← NEW: 'pending' | 'completed'
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_student_id 
            ON quizzes(student_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_date 
            ON quizzes(date)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subject 
            ON quizzes(subject)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get database connection with timeout"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    
    def _get_today_count(self, student_id: str) -> int:
        """Get count of quizzes created today by this student"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT COUNT(*) FROM quizzes
            WHERE student_id = ? AND date LIKE ?
        """, (student_id, f"{today}%"))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    # Thêm vào class QuizStorage

    def update_quiz_status(self, quiz_id: str, status: str):
        """
        Update quiz status
        
        Args:
            quiz_id: Quiz ID
            status: 'pending' | 'completed'
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE quizzes
            SET status = ?
            WHERE id = ?
        """, (status, quiz_id))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Updated quiz {quiz_id} status: {status}")

    def get_latest_pending_quiz(self, student_id: str) -> Optional[Dict]:
        """Get latest quiz with status='pending'"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM quizzes
            WHERE student_id = ? AND status = 'pending'
            ORDER BY date DESC
            LIMIT 1
        """, (student_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def save_quiz(
        self,
        student_id: str,
        content: str,
        answer_key: str,
        subject: str = None,
        topic: str = None,
        difficulty: str = None
    ) -> str:
        """Save quiz with answer key and status"""
        import uuid
        
        # FIXED VALUES
        num_questions = 10
        time_limit = 15
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        daily_count = self._get_today_count(student_id) + 1
        
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        
        # Generate UUID-based quiz ID
        quiz_id = f"quiz_{timestamp}_{unique_id}"
        
        # Insert with answer_key and status
        cursor.execute("""
            INSERT INTO quizzes (
                id, student_id, date, daily_count, content,
                subject, topic, difficulty, num_questions, time_limit,
                answer_key, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            quiz_id,
            student_id,
            now.isoformat(),
            daily_count,
            content,
            subject,
            topic,
            difficulty,
            10,
            15,
            answer_key,  # ← NEW
            'pending'    # ← NEW
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Đã lưu đề thi: {quiz_id}")
        return quiz_id
    
    def get_quiz(self, quiz_id: str) -> Optional[Dict]:
        """Get quiz by ID"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM quizzes WHERE id = ?
        """, (quiz_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_student_quizzes(
        self, 
        student_id: str, 
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict]:
        """Get recent quizzes by student (paginated)"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM quizzes
            WHERE student_id = ?
            ORDER BY date DESC
            LIMIT ? OFFSET ?
        """, (student_id, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_today_quizzes(self, student_id: str) -> List[Dict]:
        """Get all quizzes created today by student"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT * FROM quizzes
            WHERE student_id = ? AND date LIKE ?
            ORDER BY daily_count ASC
        """, (student_id, f"{today}%"))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_quizzes_by_filter(
        self,
        student_id: str = None,
        subject: str = None,
        difficulty: str = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get quizzes with filters (for API endpoints)"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build dynamic query
        query = "SELECT * FROM quizzes WHERE 1=1"
        params = []
        
        if student_id:
            query += " AND student_id = ?"
            params.append(student_id)
        
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
        
        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        
        if date_to:
            query += " AND date <= ?"
            params.append(date_to)
        
        query += " ORDER BY date DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_stats(self, student_id: str = None) -> Dict:
        """Get statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total quizzes
        if student_id:
            cursor.execute("""
                SELECT COUNT(*) FROM quizzes WHERE student_id = ?
            """, (student_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM quizzes")
        
        total = cursor.fetchone()[0]
        
        # By subject
        if student_id:
            cursor.execute("""
                SELECT subject, COUNT(*) as count
                FROM quizzes
                WHERE student_id = ? AND subject IS NOT NULL
                GROUP BY subject
            """, (student_id,))
        else:
            cursor.execute("""
                SELECT subject, COUNT(*) as count
                FROM quizzes
                WHERE subject IS NOT NULL
                GROUP BY subject
            """)
        
        by_subject = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By difficulty
        if student_id:
            cursor.execute("""
                SELECT difficulty, COUNT(*) as count
                FROM quizzes
                WHERE student_id = ? AND difficulty IS NOT NULL
                GROUP BY difficulty
            """, (student_id,))
        else:
            cursor.execute("""
                SELECT difficulty, COUNT(*) as count
                FROM quizzes
                WHERE difficulty IS NOT NULL
                GROUP BY difficulty
            """)
        
        by_difficulty = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_quizzes": total,
            "by_subject": by_subject,
            "by_difficulty": by_difficulty
        }
    
    def delete_quiz(self, quiz_id: str) -> bool:
        """Delete quiz by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def count_total(self, student_id: str = None) -> int:
        """Count total quizzes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if student_id:
            cursor.execute("""
                SELECT COUNT(*) FROM quizzes WHERE student_id = ?
            """, (student_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM quizzes")
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count