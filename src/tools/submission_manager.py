"""
tools/submission_manager.py

Quản lý việc nộp bài và chấm điểm
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional


class SubmissionManager:
    """Manage quiz submissions and grading"""
    
    def __init__(self, db_path: str = "database/quiz_storage.db"):
        self.db_path = Path(db_path)
        self._ensure_table()
    
    def _ensure_table(self):
        """Create submissions table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                quiz_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                student_answers TEXT NOT NULL,
                score REAL NOT NULL,
                daily_count INTEGER NOT NULL,
                submitted_at TEXT NOT NULL,
                duration INTEGER DEFAULT 0,
                FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_student 
            ON submissions(student_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_quiz 
            ON submissions(quiz_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_date 
            ON submissions(submitted_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get database connection with timeout"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    
    def _get_today_submission_count(self, student_id: str) -> int:
        """Count submissions today"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT COUNT(*) FROM submissions
            WHERE student_id = ? AND submitted_at LIKE ?
        """, (student_id, f"{today}%"))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def grade_submission(self, quiz_id: str, student_answers: str, answer_key: str) -> float:
        """
        Grade submission by comparing with answer key
        
        Args:
            quiz_id: Quiz ID
            student_answers: "1-A,2-B,3-C,..."
            answer_key: "1-A,2-B,3-D,..." (correct answers)
            
        Returns:
            Score (0.0 - 10.0)
        """
        try:
            def parse(ans: str) -> Dict[str, str]:
                pairs = [p.strip() for p in ans.split(",") if p.strip()]
                d = {}
                for p in pairs:
                    if "-" in p:
                        num, choice = p.split("-", 1)
                        d[num.strip()] = choice.strip().upper()
                return d
            
            student_map = parse(student_answers)
            key_map = parse(answer_key)
            
            if not key_map:
                return 0.0
            
            total = len(key_map)
            correct = 0
            for qnum, correct_choice in key_map.items():
                if student_map.get(qnum) == correct_choice:
                    correct += 1
            
            score = round((correct / total) * 10.0, 2)
            return score
        except Exception:
            return 0.0
    
    def submit_quiz(
        self,
        quiz_id: str,
        student_id: str,
        student_answers: str,
        answer_key: str
    ) -> Dict:
        """
        Submit quiz and auto-grade. Calculates duration (minutes) using quizzes.created_at.
        """
        import uuid
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Calculate daily_count
            daily_count = self._get_today_submission_count(student_id) + 1
            
            # 2. Generate submission_id with UUID
            now = datetime.now(timezone.utc)
            timestamp = now.strftime("%Y%m%d%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            
            submission_id = f"sub_{timestamp}_{unique_id}"
            
            # 3. Grade submission
            score = self.grade_submission(quiz_id, student_answers, answer_key)
            
            # 4. Calculate duration (minutes) = submitted_at - quiz.created_at
            duration_minutes = 0
            try:
                cursor.execute("SELECT created_at FROM quizzes WHERE id = ?", (quiz_id,))
                row = cursor.fetchone()
                created_at_str = row[0] if row else None

                # Debug: show what we got
                print(f"[DEBUG] quiz_id={quiz_id} created_at_str={created_at_str} now={now.isoformat()}")

                if created_at_str:
                    # parse common formats robustly
                    try:
                        if "T" in created_at_str:
                            created_at = datetime.fromisoformat(created_at_str)
                        else:
                            created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        print(f"[DEBUG] parse created_at failed: {e}; trying fromisoformat fallback")
                        created_at = datetime.fromisoformat(created_at_str)

                    # assume UTC if no tzinfo (SQLite CURRENT_TIMESTAMP is UTC)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)

                    # compute delta
                    delta = now - created_at
                    print(f"[DEBUG] delta seconds = {delta.total_seconds()}")
                    # round up to nearest minute so 17s => 1 minute
                    duration_minutes = max(0, int((delta.total_seconds() + 59) // 60))
                else:
                    print("[DEBUG] created_at not found for quiz_id")
            except Exception as e:
                print(f"[DEBUG] error computing duration: {e}")
                duration_minutes = 0
            
            # 5. Save to database (include duration)
            cursor.execute("""
                INSERT INTO submissions (
                    id, quiz_id, student_id, student_answers,
                    score, daily_count, submitted_at, duration
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                submission_id,
                quiz_id,
                student_id,
                student_answers,
                score,
                daily_count,
                now.isoformat(),
                duration_minutes
            ))
            
            conn.commit()
            
            return {
                "success": True,
                "submission_id": submission_id,
                "score": score,
                "total": 10.0,
                "percentage": (score / 10.0) * 100.0,
                "daily_count": daily_count,
                "submitted_at": now.isoformat(),
                "duration": duration_minutes
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
    
    def get_submission(self, submission_id: str) -> Optional[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, quiz_id, student_id, student_answers, score, daily_count, submitted_at, duration FROM submissions WHERE id = ?", (submission_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row[0],
            "quiz_id": row[1],
            "student_id": row[2],
            "student_answers": row[3],
            "score": row[4],
            "daily_count": row[5],
            "submitted_at": row[6],
            "duration": row[7]
        }
    
    def get_submission_with_details(self, submission_id: str, answer_key: str) -> Dict:
        """Get submission with detailed breakdown"""
        
        # Get submission
        submission = self.get_submission(submission_id)
        
        if not submission:
            return {"details": [], "correct_count": 0, "incorrect_count": 0}
        
        # Parse answers
        student_answers = submission["student_answers"]  # "1-A,2-B,..."
        
        student_dict = {}
        for item in student_answers.split(","):
            num, ans = item.strip().split("-")
            student_dict[int(num)] = ans.upper()
        
        answer_dict = {}
        for item in answer_key.split(","):
            num, ans = item.strip().split("-")
            answer_dict[int(num)] = ans.upper()
        
        # ========== DEBUG LOG ==========
        print(f"\n🔍 DEBUG - Submission details:")
        print(f"   Student answers: {student_dict}")
        print(f"   Answer key: {answer_dict}")
        # ===============================
        
        # Build details
        details = []
        correct_count = 0
        incorrect_count = 0
        
        for num in range(1, 11):  # Questions 1-10
            student_ans = student_dict.get(num, "?")
            correct_ans = answer_dict.get(num, "?")
            is_correct = (student_ans == correct_ans)
            
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
            
            details.append({
                "question_number": num,
                "student_answer": student_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct
            })
        
        # ========== DEBUG LOG ==========
        print(f"   Total details: {len(details)}")
        print(f"   Correct: {correct_count}, Incorrect: {incorrect_count}")
        # ===============================
        
        return {
            "details": details,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count
        }
    
    def get_student_submissions(
        self,
        student_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict]:
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM submissions
            WHERE student_id = ?
            ORDER BY submitted_at DESC
            LIMIT ? OFFSET ?
        """, (student_id, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def check_quiz_submitted(self, quiz_id: str, student_id: str) -> bool:
        """Check if student already submitted this quiz"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM submissions
            WHERE quiz_id = ? AND student_id = ?
        """, (quiz_id, student_id))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0