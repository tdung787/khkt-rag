"""
tools/session_manager.py

Quản lý chat sessions - tạo, xóa, list sessions
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from openai import OpenAI


class SessionManager:
    """Manage chat sessions"""
    
    def __init__(self, db_path: str = "database/quiz_storage.db", openai_client: OpenAI = None):
        self.db_path = Path(db_path)
        self.client = openai_client
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create sessions table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create chat_sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                name TEXT NOT NULL,
                first_message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                is_archived BOOLEAN DEFAULT 0
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_student 
            ON chat_sessions(student_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_updated 
            ON chat_sessions(updated_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _generate_session_id(self, student_id: str) -> str:
        """
        Generate unique session ID using UUID
        
        Format: sess_{timestamp}_{uuid4}
        Example: sess_20251112153045_a7b3c9d2
        
        Guaranteed unique, no race conditions
        """
        import uuid
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # 8 characters from UUID
        
        session_id = f"sess_{timestamp}_{unique_id}"
        
        return session_id
    
    def _generate_session_name(self, first_message: str) -> str:
        """
        Use LLM to generate session name based on first message
        
        Args:
            first_message: First message in session
            
        Returns:
            Session name (3-6 words)
        """
        if not self.client:
            # Fallback if no OpenAI client
            return first_message[:30] + ("..." if len(first_message) > 30 else "")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là trợ lý đặt tên session chat.

Nhiệm vụ: Dựa vào câu hỏi đầu tiên, đặt tên ngắn gọn cho session.

Quy tắc:
- Ngắn gọn (3-6 từ)
- Súc tích, nội dung chính
- Tiếng Việt có dấu
- KHÔNG có emoji
- KHÔNG có ký tự đặc biệt
- KHÔNG có dấu ngoặc kép

Ví dụ:
- "Giải thích định luật Newton" → "Định luật Newton"
- "Vẽ đồ thị y = x^2" → "Hàm số bậc hai"
- "Tạo đề Toán về hệ BPT" → "Hệ bất phương trình"
- "Thế nào là quang hợp?" → "Quá trình quang hợp"

Chỉ trả về TÊN, không giải thích."""
                    },
                    {
                        "role": "user",
                        "content": f'Câu hỏi: "{first_message}"'
                    }
                ],
                temperature=0.3,
                max_tokens=20
            )
            
            name = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            name = name.strip('"').strip("'")
            
            # Fallback if too long
            if len(name) > 50:
                name = name[:47] + "..."
            
            print(f"   🏷️  LLM generated name: {name}")
            
            return name
            
        except Exception as e:
            print(f"   ⚠️ Failed to generate name with LLM: {e}")
            # Fallback
            return first_message[:30] + ("..." if len(first_message) > 30 else "")
    
    def create_session(self, student_id: str, first_message: str) -> Dict:
        """
        Create new session with LLM-generated name
        
        Args:
            student_id: Student ID
            first_message: First message in session
            
        Returns:
            {
                "success": bool,
                "session": {
                    "id": "sess_xxx",
                    "name": "...",
                    "created_at": "..."
                }
            }
        """
        try:
            # Generate session ID
            session_id = self._generate_session_id(student_id)
            
            # Generate session name using LLM
            session_name = self._generate_session_name(first_message)
            
            # Create session
            now = datetime.now()
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_sessions (
                    id, student_id, name, first_message,
                    created_at, updated_at, message_count, is_archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                student_id,
                session_name,
                first_message,
                now.isoformat(),
                now.isoformat(),
                0,
                0
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Created session: {session_id} - {session_name}")
            
            return {
                "success": True,
                "session": {
                    "id": session_id,
                    "student_id": student_id,
                    "name": session_name,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "message_count": 0
                }
            }
            
        except Exception as e:
            print(f"❌ Failed to create session: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_session(self, session_id: str, student_id: str = None) -> Optional[Dict]:
        """
        Get session info with optional ownership check
        
        Args:
            session_id: Session ID
            student_id: If provided, check ownership
            
        Returns:
            Session dict or None
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if student_id:
            # With ownership check
            cursor.execute("""
                SELECT * FROM chat_sessions
                WHERE id = ? AND student_id = ?
            """, (session_id, student_id))
        else:
            # Without ownership check
            cursor.execute("""
                SELECT * FROM chat_sessions
                WHERE id = ?
            """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    
    def verify_ownership(self, session_id: str, student_id: str) -> bool:
        """
        Verify session belongs to student
        
        Returns:
            True if session belongs to student, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT student_id FROM chat_sessions
            WHERE id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return False  # Session not found
        
        return row[0] == student_id
    
    def update_session(
        self, 
        session_id: str,
        message_count: int = None,
        name: str = None
    ):
        """
        Update session metadata
        
        Args:
            session_id: Session ID
            message_count: New message count
            name: New session name
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if message_count is not None:
            updates.append("message_count = ?")
            params.append(message_count)
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        # Always update updated_at
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        
        # Add session_id to params
        params.append(session_id)
        
        query = f"""
            UPDATE chat_sessions
            SET {', '.join(updates)}
            WHERE id = ?
        """
        
        cursor.execute(query, params)
        conn.commit()
        conn.close()
    
    def delete_session(self, session_id: str, student_id: str) -> Dict:
        try:
            if not self.verify_ownership(session_id, student_id):
                return {"success": False, "error": "Session not found or doesn't belong to you"}
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Xóa messages trước (optional - CASCADE đã làm rồi)
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            
            # Xóa session
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": f"Đã xóa session {session_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_sessions(
        self,
        student_id: str,
        limit: int = 20,
        offset: int = 0,
        include_archived: bool = False
    ) -> List[Dict]:
        """
        List student's sessions
        
        Args:
            student_id: Student ID
            limit: Max number of sessions
            offset: Offset for pagination
            include_archived: Include archived sessions
            
        Returns:
            List of session dicts
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if include_archived:
            cursor.execute("""
                SELECT * FROM chat_sessions
                WHERE student_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (student_id, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM chat_sessions
                WHERE student_id = ? AND is_archived = 0
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (student_id, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def get_latest_session(self, student_id: str) -> Optional[Dict]:
        """
        Get latest session for a student (regardless of archived status)
        """
        import sqlite3
        
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT *
                FROM chat_sessions
                WHERE student_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (student_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return dict(row)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Get latest session error: {e}")
            return None
    # ============================================
    
    def archive_session(self, session_id: str, student_id: str) -> Dict:
        """
        Archive session (soft delete)
        
        Args:
            session_id: Session ID
            student_id: Student ID for ownership check
            
        Returns:
            {"success": bool}
        """
        try:
            # Verify ownership
            if not self.verify_ownership(session_id, student_id):
                return {
                    "success": False,
                    "error": "Session not found or doesn't belong to you"
                }
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE chat_sessions
                SET is_archived = 1, updated_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), session_id))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": f"Đã archive session {session_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }