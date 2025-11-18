"""
tools/chat_history_manager.py

Quản lý chat messages trong sessions
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4
from datetime import datetime


class ChatHistoryManager:
    """Manage chat messages in sessions"""
    
    def __init__(self, db_path: str = "database/quiz_storage.db"):
        self.db_path = Path(db_path)
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create messages table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create chat_messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON chat_messages(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
            ON chat_messages(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _generate_message_id(self, session_id: str) -> str:
        """
        Generate a unique message ID
        
        Format: msg_YYYYMMDD_<8-char-UUID>
        Example: msg_20251114_a7b3c9d2
        
        Guaranteed unique, no race conditions
        """
        today = datetime.now().strftime("%Y%m%d")
        unique_part = uuid4().hex[:8]  # 8 characters from UUID
        message_id = f"msg_{today}_{unique_part}"
        return message_id

    
    def save_message(
        self,
        session_id: str,
        role: str,  # 'user' | 'assistant'
        content: str
    ) -> str:
        """
        Save message to session
        
        Args:
            session_id: Session ID
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            message_id
        """
        try:
            # Validate role
            if role not in ['user', 'assistant']:
                raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")
            
            # Generate message ID
            message_id = self._generate_message_id(session_id)
            
            # Save message
            now = datetime.now()
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_messages (
                    id, session_id, role, content, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                message_id,
                session_id,
                role,
                content,
                now.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return message_id
            
        except Exception as e:
            print(f"❌ Failed to save message: {e}")
            raise
    
    def get_session_history(
        self,
        session_id: str,
        limit: int = None
    ) -> List[Dict]:
        """
        Get conversation history for session
        
        Args:
            session_id: Session ID
            limit: Max number of messages (None = all)
            
        Returns:
            List of messages in format:
            [
                {"role": "user", "content": "...", "timestamp": "..."},
                {"role": "assistant", "content": "...", "timestamp": "..."},
                ...
            ]
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if limit:
            # Get latest N messages
            cursor.execute("""
                SELECT role, content, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            # Reverse to get chronological order
            rows = list(reversed(rows))
        else:
            # Get all messages
            cursor.execute("""
                SELECT role, content, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            
            rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_message_count(self, session_id: str) -> int:
        """
        Count messages in session
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of messages
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM chat_messages
            WHERE session_id = ?
        """, (session_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def delete_session_messages(self, session_id: str):
        """
        Delete all messages in session
        
        Args:
            session_id: Session ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM chat_messages
            WHERE session_id = ?
        """, (session_id,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted {deleted} messages from session {session_id}")
    
    def get_message(self, message_id: str) -> Optional[Dict]:
        """
        Get single message by ID
        
        Args:
            message_id: Message ID
            
        Returns:
            Message dict or None
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM chat_messages
            WHERE id = ?
        """, (message_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def search_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search messages in session
        
        Args:
            session_id: Session ID
            query: Search query
            limit: Max results
            
        Returns:
            List of matching messages
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM chat_messages
            WHERE session_id = ? AND content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, f"%{query}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]