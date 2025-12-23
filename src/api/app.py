"""
FastAPI application for Quiz Management System

Provides simple REST API for accessing quiz history
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import sys
import os
import time
import uuid
import io
import json
import subprocess
import re
import sqlite3
import requests
import tempfile
import shutil
import traceback
import threading
import queue
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import base64
from PIL import Image
from datetime import datetime
from fastapi.staticfiles import StaticFiles

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.quiz_storage import QuizStorage
from src.tools.submission_manager import SubmissionManager
from query import ScienceQASystem, QuestionRetriever, IntentClassifier, SimpleAgent
from src.tools.session_manager import SessionManager
from src.tools.chat_history_manager import ChatHistoryManager
from src.tools.evaluation_storage import EvaluationStorage

# ==================== EXTERNAL API CONFIG ====================
evaluation_storage = EvaluationStorage()
EXTERNAL_API_BASE_URL = os.getenv("EXTERNAL_API_BASE_URL", "https://v5bfv7qs-3001.asse.devtunnels.ms")

# ==================== HELPER FUNCTIONS ====================

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Quiz Management API",
    description="API để quản lý đề kiểm tra trắc nghiệm",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (production nên restrict)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
chat_images_dir = Path("database/chat_images")
chat_images_dir.mkdir(parents=True, exist_ok=True)

app.mount(
    "/static/images", 
    StaticFiles(directory=str(chat_images_dir)), 
    name="chat_images"
)
print(f"✅ Static images mounted at: /static/images")

# ==================== INITIALIZE COMPONENTS ====================
# Initialize storage
storage = QuizStorage()
submission_manager = SubmissionManager()

# ========== INIT OPENAI CLIENT (ONLY ONCE) ==========
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
session_manager = SessionManager(openai_client=openai_client)
chat_history_manager = ChatHistoryManager()
print("✅ Session managers initialized")

# ========== GLOBAL STORAGE ==========
# Store temp files: {file_id: (pdf_path, created_time)}
PDF_STORAGE = {}
PDF_STORAGE_DIR = Path("temp_pdfs")
PDF_STORAGE_DIR.mkdir(exist_ok=True)

# ========== CLEANUP OLD FILES ==========
def cleanup_old_pdfs():
    """Delete PDFs older than 1 hour"""
    current_time = time.time()
    to_delete = []
    
    for file_id, (pdf_path, created_time) in PDF_STORAGE.items():
        if current_time - created_time > 3600:  # 1 hour
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                to_delete.append(file_id)
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
    
    for file_id in to_delete:
        del PDF_STORAGE[file_id]

# ========== CLEANUP FUNCTION ==========

def cleanup_temp_dir(temp_dir: str):
    """Delete temp directory after response is sent"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🗑️  Cleaned up: {temp_dir}")
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")

def preprocess_markdown(content: str) -> str:
    """Fix [ ] and ( ) to $ $ and $$ $$"""
    
    # Replace standalone [ ... ] with $$ ... $$
    content = re.sub(
        r'\[\s*\n((?:[^\]]|\n)+?)\n\s*\]',
        r'$$\1$$',
        content,
        flags=re.DOTALL
    )
    
    # Replace inline ( ... ) with $ ... $
    content = re.sub(
        r'\(\s*([^)]*\\[^)]*?)\s*\)',
        r'$\1$',
        content
    )
    
    return content


def markdown_text_to_pdf(
    markdown_content: str,
    output_filename: str = "output.pdf",
    toc: bool = False,
    number_sections: bool = False,
    clean_markdown: bool = True
) -> str:
    """
    Convert markdown text to PDF
    
    Args:
        markdown_content: Markdown string
        output_filename: Output PDF filename
        toc: Include table of contents
        number_sections: Auto-number sections
        clean_markdown: Preprocess markdown to fix LaTeX
    
    Returns:
        Path to generated PDF
    """
    
    # Check dependencies
    if not shutil.which('pandoc'):
        raise RuntimeError("Pandoc not installed")
    
    if not shutil.which('xelatex'):
        raise RuntimeError("XeLaTeX not installed")
    
    # Preprocess if enabled
    if clean_markdown:
        markdown_content = preprocess_markdown(markdown_content)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Write markdown to temp file
        temp_md = os.path.join(temp_dir, "input.md")
        with open(temp_md, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Output path
        output_pdf = os.path.join(temp_dir, output_filename)
        
        # Build pandoc command
        cmd = [
            'pandoc',
            temp_md,
            '-o', output_pdf,
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=12pt',
            '-V', 'documentclass=article',
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=blue',
            '-V', 'urlcolor=blue',
            '--highlight-style=tango',
            '--standalone'
        ]
        
        if toc:
            cmd.extend(['--toc', '--toc-depth=3'])
        
        if number_sections:
            cmd.append('--number-sections')
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if PDF was created
        if not os.path.exists(output_pdf):
            raise RuntimeError("PDF was not created")
        
        return output_pdf
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Pandoc error: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")

# ========== INIT RAG COMPONENTS (SINGLETON) ==========
try:
    intent_classifier = IntentClassifier(openai_client)
    # retriever = QuestionRetriever(
    #     openai_client, 
    #     "database/qdrant_storage", 
    #     "KHTN_QA"
    # )
    retriever = "initialized"
    print("✅ Shared RAG components initialized")
except Exception as e:
    print(f"⚠️ Failed to init RAG components: {e}")
    import traceback
    traceback.print_exc()
    intent_classifier = None
    retriever = None

# ==================== HEALTH CHECK ====================
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Quiz Management API is running",
        "endpoints": {
            "latest": "/api/quiz/latest",
            "all": "/api/quiz/all",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    """Detailed health check"""
    total = storage.count_total()
    return {
        "status": "healthy",
        "database": "connected",
        "total_quizzes": total
    }


# ==================== API 1: LATEST QUIZ ====================
@app.get("/api/quiz/latest")
def get_latest_quiz(
    student_id: Optional[str] = Query(None, description="Student ID to filter by")
) -> Dict:
    """
    Lấy bài kiểm tra mới nhất
    
    Args:
        student_id: Optional - Lọc theo student ID
        
    Returns:
        Bài kiểm tra mới nhất hoặc error
    """
    try:
        if student_id:
            # Get latest quiz for specific student
            quizzes = storage.get_student_quizzes(student_id, limit=1, offset=0)
            
            if not quizzes:
                return {
                    "success": False,
                    "message": f"Không tìm thấy đề kiểm tra cho student_id: {student_id}"
                }
            
            return {
                "success": True,
                "data": quizzes[0]
            }
        else:
            # Get latest quiz overall
            quizzes = storage.get_quizzes_by_filter(limit=1, offset=0)
            
            if not quizzes:
                return {
                    "success": False,
                    "message": "Chưa có đề kiểm tra nào trong hệ thống"
                }
            
            return {
                "success": True,
                "data": quizzes[0]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ==================== API 2: ALL QUIZZES ====================
@app.get("/api/quiz/all")
def get_all_quizzes(
    student_id: Optional[str] = Query(None, description="Filter by student ID"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format: 2025-01-01)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format: 2025-01-31)")
) -> Dict:
    """
    Get all quizzes with filters (no pagination)
    
    Args:
        student_id: Optional - Filter by student
        subject: Optional - Filter by subject
        difficulty: Optional - Filter by difficulty
        date_from: Optional - Filter from date
        date_to: Optional - Filter to date
        
    Returns:
        List of all quizzes matching filters
    """
    try:
        # Get filtered quizzes (no limit)
        quizzes = storage.get_quizzes_by_filter(
            student_id=student_id,
            subject=subject,
            difficulty=difficulty,
            date_from=date_from,
            date_to=date_to,
            limit=999999,  # Get all
            offset=0
        )
        
        return {
            "success": True,
            "total": len(quizzes),
            "filters": {
                "student_id": student_id,
                "subject": subject,
                "difficulty": difficulty,
                "date_from": date_from,
                "date_to": date_to
            },
            "data": quizzes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ==================== SUBMISSION ====================
    
@app.get("/api/submission/all")
def get_all_submissions(
    student_id: Optional[str] = Query(None, description="Filter by student ID"),
    quiz_id: Optional[str] = Query(None, description="Filter by quiz ID"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format: 2025-01-01)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format: 2025-01-31)"),
) -> Dict:
    """
    Get all submissions with filters (no pagination)
    
    Returns only basic fields: id, quiz_id, student_id, student_answers, 
    score, daily_count, submitted_at, duration
    """
    try:
        import sqlite3
        
        conn = submission_manager._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM submissions WHERE 1=1"
        params = []
        
        # Add filters
        if student_id:
            query += " AND student_id = ?"
            params.append(student_id)
        
        if quiz_id:
            query += " AND quiz_id = ?"
            params.append(quiz_id)
        
        if date_from:
            query += " AND submitted_at >= ?"
            params.append(date_from)
        
        if date_to:
            query += " AND submitted_at < date(?, '+1 day')"
            params.append(date_to)
        
        # Order by date DESC
        query += " ORDER BY submitted_at DESC"
        
        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Format results
        submissions = [dict(row) for row in rows]
        
        conn.close()
        
        return {
            "success": True,
            "total": len(submissions),
            "filters": {
                "student_id": student_id,
                "quiz_id": quiz_id,
                "date_from": date_from,
                "date_to": date_to,
            },
            "data": submissions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
# ==================== DAILY COUNT ====================
@app.get("/api/quiz/daily-count")
def get_daily_count(
    student_id: str = Query(..., description="Student ID (required)")
) -> Dict:
    """
    Đếm số lần học sinh làm bài theo từng ngày
    
    Args:
        student_id: Student ID (bắt buộc)
        
    Returns:
        Thống kê số bài theo ngày
    """
    try:
        # Get all quizzes of student
        all_quizzes = storage.get_student_quizzes(student_id, limit=9999, offset=0)
        
        # Group by date
        daily_stats = {}
        for quiz in all_quizzes:
            date = quiz["date"].split("T")[0]  # Extract YYYY-MM-DD
            
            if date not in daily_stats:
                daily_stats[date] = {
                    "date": date,
                    "count": 0,
                    "daily_counts": [],
                    "subjects": []
                }
            
            daily_stats[date]["count"] += 1
            daily_stats[date]["daily_counts"].append(quiz["daily_count"])
            daily_stats[date]["subjects"].append(quiz.get("subject"))
        
        # Convert to list and sort by date descending
        daily_list = sorted(
            daily_stats.values(), 
            key=lambda x: x["date"], 
            reverse=True
        )
        
        # Calculate summary
        today_date = datetime.now().strftime("%Y-%m-%d")
        today_count = daily_stats.get(today_date, {}).get("count", 0)
        
        return {
            "success": True,
            "student_id": student_id,
            "total_days": len(daily_list),
            "total_quizzes": len(all_quizzes),
            "today": {
                "date": today_date,
                "count": today_count
            },
            "daily_breakdown": daily_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/quiz/by-date")
def get_quizzes_by_date(
    student_id: str = Query(..., description="Student ID (required)"),
    date: str = Query(..., description="Date in YYYY-MM-DD format (e.g., 2025-01-10)")
) -> Dict:
    """
    Lấy tất cả bài kiểm tra của 1 ngày cụ thể
    
    Args:
        student_id: Student ID (bắt buộc)
        date: Ngày cần lấy (YYYY-MM-DD)
        
    Returns:
        Danh sách quiz của ngày đó
    """
    try:
        # Get all quizzes and filter by date
        all_quizzes = storage.get_student_quizzes(student_id, limit=9999, offset=0)
        
        quizzes_on_date = [
            q for q in all_quizzes 
            if q["date"].startswith(date)
        ]
        
        # Sort by daily_count
        quizzes_on_date.sort(key=lambda x: x["daily_count"])
        
        return {
            "success": True,
            "date": date,
            "student_id": student_id,
            "count": len(quizzes_on_date),
            "data": quizzes_on_date
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
# ==================== SESSION MANAGEMENT ENDPOINTS ====================

@app.post("/api/session/create")
def create_session(
    student_id: str = Query(..., description="Student ID (required)"),
    first_message: Optional[str] = Query(None, description="Optional first message to start conversation")
) -> Dict:
    """
    Create new chat session (or reuse empty one)
    
    Logic:
    - If latest session is empty (message_count=0) → Reuse it
    - Otherwise → Create new session
    
    - Validates student_id against external API
    - If first_message provided: Create/reuse session + process message + get response
    - If no first_message: Create/reuse empty session with default name
    """
    try:
        # ========== CHECK FOR EMPTY SESSION ==========
        latest_session = session_manager.get_latest_session(student_id)
        
        if latest_session and latest_session.get('message_count', 0) == 0:
            print(f"   ♻️  Reusing empty session: {latest_session['id']}")
            existing_session_id = latest_session['id']
        else:
            existing_session_id = None
            print(f"   ✨ Will create new session")
        # =============================================
        
        # ========== CASE 1: WITH FIRST MESSAGE ==========
        if first_message:
            # Reuse or create session
            if existing_session_id:
                session_id = existing_session_id
                
                # Update session name based on first message
                new_name = session_manager._generate_session_name(first_message)
                
                conn = session_manager._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE chat_sessions
                    SET name = ?, first_message = ?, updated_at = ?
                    WHERE id = ?
                """, (new_name, first_message, datetime.now().isoformat(), session_id))
                conn.commit()
                conn.close()
                
                session = session_manager.get_session(session_id, student_id)
                print(f"   ♻️  Reused and renamed session: {session_id} - {new_name}")
            else:
                # Create new session
                session_result = session_manager.create_session(
                    student_id=student_id,
                    first_message=first_message
                )
                
                if not session_result["success"]:
                    raise HTTPException(
                        status_code=500,
                        detail=session_result.get("error", "Failed to create session")
                    )
                
                session = session_result["session"]
                print(f"   ✨ Created new session: {session['id']} - {session['name']}")
            
            # NOW create agent with session's student_id
            if not openai_client or not intent_classifier:
                raise HTTPException(
                    status_code=503,
                    detail="RAG components not initialized"
                )
            
            try:
                agent = SimpleAgent(
                    openai_client,
                    intent_classifier,
                    retriever,
                    session['student_id']
                )
                print(f"   ✅ Agent initialized")
            except Exception as e:
                print(f"   ❌ Agent init error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize agent: {str(e)}"
                )
            
            # Process first message and get response
            response = agent.query(first_message, conversation_history=[])
            
            # Save messages to session
            try:
                # Save user message
                chat_history_manager.save_message(
                    session_id=session['id'],
                    role="user",
                    content=first_message
                )
                
                # Save assistant response
                chat_history_manager.save_message(
                    session_id=session['id'],
                    role="assistant",
                    content=response
                )
                
                # Update message count
                new_count = chat_history_manager.get_message_count(session['id'])
                session_manager.update_session(
                    session_id=session['id'],
                    message_count=new_count
                )
                
                print(f"   💾 Saved initial messages (total: {new_count})")
                
            except Exception as e:
                print(f"   ⚠️ Failed to save messages: {e}")
            
            return {
                "success": True,
                "session": session,
                "response": response,
                "has_first_message": True,
                "reused_session": existing_session_id is not None
            }
        
        # ========== CASE 2: EMPTY SESSION ==========
        else:
            # Reuse or create empty session
            if existing_session_id:
                session = session_manager.get_session(existing_session_id, student_id)
                print(f"   ♻️  Reusing empty session: {existing_session_id}")
                
                return {
                    "success": True,
                    "session": session,
                    "response": None,
                    "has_first_message": False,
                    "reused_session": True
                }
            else:
                # Create new empty session
                session_id = session_manager._generate_session_id(student_id)
                default_name = "Cuộc trò chuyện mới"
                
                now = datetime.now()
                
                conn = session_manager._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO chat_sessions (
                        id, student_id, name, first_message,
                        created_at, updated_at, message_count, is_archived
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    student_id,
                    default_name,
                    "",
                    now.isoformat(),
                    now.isoformat(),
                    0,
                    0
                ))
                
                conn.commit()
                conn.close()
                
                print(f"   ✨ Created empty session: {session_id}")
                
                return {
                    "success": True,
                    "session": {
                        "id": session_id,
                        "student_id": student_id,
                        "name": default_name,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                        "message_count": 0
                    },
                    "response": None,
                    "has_first_message": False,
                    "reused_session": False
                }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation error: {str(e)}")

@app.get("/api/session/list")
def list_sessions(
    student_id: str = Query(..., description="Student ID (required)"),
    limit: int = Query(20, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    include_archived: bool = Query(False, description="Include archived sessions")
) -> Dict:
    """
    List all sessions for a student
    
    Args:
        student_id: Student ID
        limit: Max sessions to return
        offset: Pagination offset
        include_archived: Include archived sessions
        
    Returns:
        List of sessions
    """
    try:
        sessions = session_manager.list_sessions(
            student_id=student_id,
            limit=limit,
            offset=offset,
            include_archived=include_archived
        )
        
        return {
            "success": True,
            "student_id": student_id,
            "count": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List sessions error: {str(e)}")


@app.get("/api/session")
def get_session(
    session_id: str = Query(..., description="Session ID"),
    student_id: str = Query(..., description="Student ID for ownership verification")
) -> Dict:
    """
    Get session info with full conversation history
    
    Args:
        session_id: Session ID (query param)
        student_id: Student ID for verification
        
    Returns:
        Session info + full chat history in user-chatbot pairs
    """
    try:
        # Verify ownership
        if not session_manager.verify_ownership(session_id, student_id):
            raise HTTPException(
                status_code=403,
                detail="Session not found or doesn't belong to you"
            )
        
        session = session_manager.get_session(session_id, student_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get full chat history
        messages = chat_history_manager.get_session_history(session_id)
        
        # Format messages into conversation pairs
        conversation = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                # Complete pair (user + assistant)
                conversation.append({
                    "user": {
                        "content": messages[i]["content"],
                        "timestamp": messages[i]["timestamp"]
                    },
                    "chatbot": {
                        "content": messages[i + 1]["content"],
                        "timestamp": messages[i + 1]["timestamp"]
                    }
                })
            else:
                # Odd message (user only, no response yet)
                conversation.append({
                    "user": {
                        "content": messages[i]["content"],
                        "timestamp": messages[i]["timestamp"]
                    },
                    "chatbot": None
                })
        
        return {
            "success": True,
            "session": {
                "id": session['id'],
                "name": session['name'],
                "student_id": session['student_id'],
                "created_at": session['created_at'],
                "updated_at": session['updated_at'],
                "message_count": session['message_count'],
                "is_archived": session['is_archived']
            },
            "conversation": conversation,
            "total_pairs": len(conversation)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get session error: {str(e)}")


@app.get("/api/session/{session_id}/history")
def get_session_history(
    session_id: str,
    student_id: str = Query(..., description="Student ID for ownership verification"),
    limit: Optional[int] = Query(None, description="Limit number of messages")
) -> Dict:
    """
    Get chat history for a session
    
    Args:
        session_id: Session ID
        student_id: Student ID for verification
        limit: Optional limit on number of messages
        
    Returns:
        Session info + chat history
    """
    try:
        # Verify ownership
        if not session_manager.verify_ownership(session_id, student_id):
            raise HTTPException(
                status_code=403,
                detail="Session not found or doesn't belong to you"
            )
        
        # Get session info
        session = session_manager.get_session(session_id, student_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get chat history
        messages = chat_history_manager.get_session_history(
            session_id=session_id,
            limit=limit
        )
        
        return {
            "success": True,
            "session": {
                "id": session['id'],
                "name": session['name'],
                "student_id": session['student_id'],
                "created_at": session['created_at'],
                "updated_at": session['updated_at'],
                "message_count": session['message_count']
            },
            "messages": messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get history error: {str(e)}")


@app.delete("/api/session")
def delete_session(
    session_id: str = Query(..., description="Session ID"),
    student_id: str = Query(..., description="Student ID for ownership verification")
) -> Dict:
    """
    Delete a session and all its messages
    
    Args:
        session_id: Session ID to delete (query param)
        student_id: Student ID for verification
        
    Returns:
        Success message
    """
    try:
        # Delete session (will also delete messages via CASCADE)
        result = session_manager.delete_session(session_id, student_id)
        
        if not result["success"]:
            if "doesn't belong to you" in result.get("error", ""):
                raise HTTPException(status_code=403, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": result["message"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete session error: {str(e)}")


@app.put("/api/session/{session_id}/rename")
def rename_session(
    session_id: str,
    student_id: str = Query(..., description="Student ID for ownership verification"),
    new_name: str = Query(..., description="New session name")
) -> Dict:
    """
    Rename a session
    
    Args:
        session_id: Session ID
        student_id: Student ID for verification
        new_name: New name for session
        
    Returns:
        Success message
    """
    try:
        # Verify ownership
        if not session_manager.verify_ownership(session_id, student_id):
            raise HTTPException(
                status_code=403,
                detail="Session not found or doesn't belong to you"
            )
        
        # Update session name
        session_manager.update_session(
            session_id=session_id,
            name=new_name
        )
        
        return {
            "success": True,
            "message": f"Session renamed to: {new_name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rename session error: {str(e)}")


@app.post("/api/session/{session_id}/archive")
def archive_session(
    session_id: str,
    student_id: str = Query(..., description="Student ID for ownership verification")
) -> Dict:
    """
    Archive a session (soft delete)
    
    Args:
        session_id: Session ID
        student_id: Student ID for verification
        
    Returns:
        Success message
    """
    try:
        result = session_manager.archive_session(session_id, student_id)
        
        if not result["success"]:
            if "doesn't belong to you" in result.get("error", ""):
                raise HTTPException(status_code=403, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "message": result["message"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archive session error: {str(e)}")
    
# ==================== SUBMISSION ENDPOINTS ====================

@app.get("/api/quiz/current-status")
def get_current_quiz_status(
    student_id: str = Query(..., description="Student ID (required)")
) -> Dict:
    """
    Check if student has pending quiz
    
    Returns:
        Quiz info if pending, or null
    """
    try:
        pending_quiz = storage.get_latest_pending_quiz(student_id)
        
        if pending_quiz:
            return {
                "success": True,
                "has_pending": True,
                "quiz": {
                    "id": pending_quiz["id"],
                    "subject": pending_quiz.get("subject"),
                    "topic": pending_quiz.get("topic"),
                    "difficulty": pending_quiz.get("difficulty"),
                    "created_at": pending_quiz.get("date")
                }
            }
        else:
            return {
                "success": True,
                "has_pending": False,
                "quiz": None
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/submission/submit")
def submit_quiz(
    quiz_id: str = Query(..., description="Quiz ID"),
    student_id: str = Query(..., description="Student ID"),
    answers: str = Query(..., description="Student answers in format: 1-A,2-B,3-C,...")
) -> Dict:
    """
    Submit quiz and auto-grade
    
    Args:
        quiz_id: Quiz ID to submit
        student_id: Student ID
        answers: Format "1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B"
        
    Returns:
        Submission result with score
    """
    try:
        # 1. Check if quiz exists
        quiz = storage.get_quiz(quiz_id)
        if not quiz:
            raise HTTPException(status_code=404, detail=f"Quiz not found: {quiz_id}")
        
        # 2. Check if quiz belongs to student
        if quiz["student_id"] != student_id:
            raise HTTPException(status_code=403, detail="Quiz does not belong to this student")
        
        # 3. Check if quiz is pending
        if quiz.get("status") != "pending":
            raise HTTPException(status_code=400, detail="Quiz already submitted")
        
        # 4. Check if already submitted
        if submission_manager.check_quiz_submitted(quiz_id, student_id):
            raise HTTPException(status_code=400, detail="Quiz already submitted")
        
        # 5. Validate answers format
        if not answers or len(answers.split(',')) != 10:
            raise HTTPException(status_code=400, detail="Answers must have exactly 10 items (1-A,2-B,...)")
        
        # 6. Get answer key from quiz
        answer_key = quiz.get("answer_key")
        if not answer_key:
            raise HTTPException(status_code=500, detail="Quiz missing answer key")
        
        # 7. Submit and grade
        result = submission_manager.submit_quiz(
            quiz_id=quiz_id,
            student_id=student_id,
            student_answers=answers,
            answer_key=answer_key
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Submission failed"))
        
        # 8. Update quiz status to completed
        storage.update_quiz_status(quiz_id, "completed")
        
        return {
            "success": True,
            "message": "Đã nộp bài và chấm điểm thành công!",
            "submission_id": result["submission_id"],
            "score": result["score"],
            "total": result["total"],
            "percentage": result["percentage"],
            "daily_count": result["daily_count"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/submission/{submission_id}")
def get_submission(submission_id: str) -> Dict:
    """Get submission by ID (basic info)"""
    try:
        submission = submission_manager.get_submission(submission_id)
        
        if not submission:
            raise HTTPException(status_code=404, detail=f"Submission not found: {submission_id}")
        
        return {
            "success": True,
            "data": submission
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/submission/{submission_id}/result")
def get_submission_result(submission_id: str) -> Dict:
    """
    Get detailed submission result with correct/incorrect breakdown
    """
    try:
        # Get submission
        submission = submission_manager.get_submission(submission_id)
        
        if not submission:
            raise HTTPException(status_code=404, detail=f"Submission not found: {submission_id}")
        
        # Get quiz to get answer key
        quiz = storage.get_quiz(submission["quiz_id"])
        
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")
        
        # Get detailed result
        detailed = submission_manager.get_submission_with_details(
            submission_id,
            quiz["answer_key"]
        )
        
        return {
            "success": True,
            "submission_id": submission_id,
            "quiz_id": submission["quiz_id"],
            "student_id": submission["student_id"],
            "score": submission["score"],
            "total": 10.0,
            "percentage": (submission["score"] / 10.0) * 100,
            "correct_count": detailed["correct_count"],
            "incorrect_count": detailed["incorrect_count"],
            "submitted_at": submission["submitted_at"],
            "daily_count": submission["daily_count"],
            "details": detailed["details"],
            "quiz_info": {
                "subject": quiz.get("subject"),
                "topic": quiz.get("topic"),
                "difficulty": quiz.get("difficulty")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/submission/student/{student_id}")
def get_student_submissions(
    student_id: str,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> Dict:
    """Get submission history for a student"""
    try:
        submissions = submission_manager.get_student_submissions(
            student_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "student_id": student_id,
            "count": len(submissions),
            "data": submissions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ==================== BONUS: SINGLE QUIZ BY ID ====================
@app.get("/api/quiz/{quiz_id}")
def get_quiz_by_id(quiz_id: str) -> Dict:
    """
    Lấy chi tiết 1 bài kiểm tra theo ID
    
    Args:
        quiz_id: Quiz ID (e.g., quiz_20250110_001)
        
    Returns:
        Quiz details
    """
    try:
        quiz = storage.get_quiz(quiz_id)
        
        if not quiz:
            raise HTTPException(status_code=404, detail=f"Quiz not found: {quiz_id}")
        
        return {
            "success": True,
            "data": quiz
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ==================== BONUS: STATISTICS ====================
@app.get("/api/stats")
def get_statistics(
    student_id: Optional[str] = Query(None, description="Get stats for specific student")
) -> Dict:
    """
    Lấy thống kê
    
    Args:
        student_id: Optional - Stats for specific student
        
    Returns:
        Statistics data
    """
    try:
        stats = storage.get_stats(student_id=student_id)
        
        return {
            "success": True,
            "student_id": student_id,
            "data": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ==================== RAG QUERY ENDPOINT ====================
@app.post("/api/rag/query")
async def rag_query(
    user_input: str = Form(..., description="User question or request"),
    session_id: str = Form(..., description="Session ID (required)"),
    student_id: Optional[str] = Form(None, description="Optional student ID for verification"),
    image: Optional[UploadFile] = File(None, description="Optional image file (quiz/problem)")
) -> Dict:
    """
    Query the RAG system within a session with optional image support
    
    User must create a session first via POST /api/session/create
    
    Supports:
    - Answering questions about subjects
    - Creating quizzes
    - Drawing graphs
    - Submitting answers
    - Image-based questions (solving problems from photos)
    - General Q&A
    
    All interactions are saved to the session's chat history.
    
    Args:
        user_input: User's question or command
        session_id: Session ID (REQUIRED)
        student_id: Optional student ID for ownership verification
        image: Optional image file (JPEG/PNG)
        
    Returns:
        RAG system response with session info
    """
        # ========== INITIALIZE VARIABLES ==========
    qdrant_client = None
    agent = None
    try:
        # ========== VALIDATE SESSION ==========
        session = session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        # Verify ownership if student_id provided
        if student_id:
            if not session_manager.verify_ownership(session_id, student_id):
                raise HTTPException(
                    status_code=403,
                    detail="Session doesn't belong to you"
                )
        
        print(f"   📂 Using session: {session_id} - {session.get('name')}")
        
        # ========== PROCESS IMAGE IF PROVIDED ==========
        image_context = None
        image_url = None

        if image:
            print(f"   🖼️  Image received: {image.filename}")
            try:
                from datetime import datetime
                import uuid
                
                # Read image
                image_data = await image.read()
                
                # Open and convert to RGB (handle PNG/RGBA)
                img = Image.open(io.BytesIO(image_data))
                
                # Convert RGBA/LA/P to RGB BEFORE any processing
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                    print(f"   🔄 Converted {img.mode} to RGB")
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                    print(f"   🔄 Converted {img.mode} to RGB")
                
                # Calculate resize ratio
                # Resize to 1024px for better quality
                max_size = 1024
                ratio = min(max_size / img.width, max_size / img.height)
                if ratio < 1:
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    print(f"   📐 Resized: original → {img.width}x{img.height}")
                
                # At this point, img is GUARANTEED to be RGB mode
                
                # ========== SAVE IMAGE TO DISK ==========
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                filename = f"chat_img_{session_id}_{timestamp}_{unique_id}.jpg"
                
                # Save to disk (local directory)
                filepath = f"database/chat_images/{filename}"
                img.save(filepath, format="JPEG", quality=95)  # Now safe to save as JPEG
                
                # Generate public URL
                api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8110')
                image_url = f"{api_base_url}/static/images/{filename}"
                print(f"   💾 Saved image: {filepath}")
                print(f"   🌐 Public URL: {image_url}")
                # ========================================
                
                # Convert to base64 for LLM (use the same RGB image)
                buffer_base64 = io.BytesIO()
                img.save(buffer_base64, format="JPEG", quality=95)
                base64_image = base64.b64encode(buffer_base64.getvalue()).decode()
                
                image_context = {
                    "base64": base64_image,
                    "filename": image.filename,
                    "size": f"{img.width}x{img.height}",
                    "url": image_url  # ← THÊM URL
                }
                
                print(f"   ✅ Image processed: {img.width}x{img.height}, {len(base64_image)/1024:.1f}KB")
                
            except Exception as e:
                print(f"   ⚠️ Image processing failed: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file: {str(e)}"
                )
        # ===============================================
        
        # ========== INITIALIZE RAG SYSTEM WITH STUDENT_ID ==========
        session_student_id = session.get('student_id')
        
        # Check if shared components available
        if not openai_client or not intent_classifier:
            raise HTTPException(
                status_code=503,
                detail="RAG components not initialized"
            )

        # Create lightweight agent instance with fresh retriever
        agent = None
        qdrant_client = None
        try:

            # Create fresh retriever
            fresh_retriever = QuestionRetriever(
                openai_client,
                "database/qdrant_storage",  # Pass client directly
                "KHTN_QA"
            )
            
            agent = SimpleAgent(
                openai_client,
                intent_classifier,
                fresh_retriever,  # Use fresh retriever
                session_student_id
            )
            print(f"   ✅ Agent initialized for student: {session_student_id}")
        except Exception as e:
            print(f"   ❌ RAG init error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize RAG system: {str(e)}"
            )
        # ===========================================================
        
        # ========== LOAD CONVERSATION HISTORY ==========
        conversation_history = chat_history_manager.get_session_history(session_id)
        print(f"   📜 Loaded {len(conversation_history)} messages from history")
        
        # ========== PROCESS QUERY WITH IMAGE ==========
        result = agent.query(  # ← ĐỔI response THÀNH result
            user_input, 
            conversation_history,
            image_context=image_context
        )
        
        response = result["response"]  # ← THÊM DÒNG NÀY
        final_query = result["final_query"]  # ← THÊM DÒNG NÀY
        
        print(f"   📝 User input: {user_input[:50]}...")
        print(f"   📝 Final query: {final_query[:50]}...")  # ← DEBUG
        
        # ========== SAVE TO SESSION ==========
                # ========== SAVE TO SESSION ==========
        try:
            if image_context:
                user_content = f"""![Uploaded image]({image_context['url']})

{user_input}

<!-- EXTRACTED_TEXT
{final_query}
-->"""
                print(f"   💾 Saving with hidden extracted text ({len(final_query)} chars)")
            else:
                user_content = final_query
                print(f"   💾 Saving text input: {len(final_query)} chars")
            
            chat_history_manager.save_message(
                session_id=session_id,
                role="user",
                content=user_content
            )
            # ===========================================================
            
            # Save assistant response
            chat_history_manager.save_message(
                session_id=session_id,
                role="assistant",
                content=response
            )
            
            # Update session metadata
            new_count = chat_history_manager.get_message_count(session_id)
            
            # ========== AUTO RENAME IF EMPTY SESSION ==========
            if session.get('first_message') == "" and new_count == 2:
                new_name = session_manager._generate_session_name(user_input)
                
                session_manager.update_session(
                    session_id=session_id,
                    message_count=new_count,
                    name=new_name
                )
                
                conn = session_manager._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE chat_sessions
                    SET first_message = ?
                    WHERE id = ?
                """, (user_input, session_id))
                conn.commit()
                conn.close()
                
                print(f"   🏷️  Auto-renamed empty session to: '{new_name}'")
            else:
                session_manager.update_session(
                    session_id=session_id,
                    message_count=new_count
                )
            # ================================================
            
            print(f"   💾 Saved messages to session (total: {new_count})")
            
        except Exception as e:
            print(f"   ⚠️ Failed to save to session: {e}")
        
        # ========== RETURN RESPONSE ==========
        
        full_user_message = user_input
        if image_context:
            image_markdown = f"![Uploaded image]({image_context['url']})"
            full_user_message = f"{image_markdown}\n\n{user_input}"
            
        return {
            "success": True,
            "user_input": full_user_message,
            "has_image": image_context is not None,
            "session": {
                "id": session['id'],
                "name": session.get('name'),
                "student_id": session.get('student_id'),
                "message_count": new_count
            },
            "response": response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query error: {str(e)}")
    finally:  # ← THÊM DÒNG NÀY
        # Close Qdrant connection
        import gc
        gc.collect()
    
# ==================== STUDENT EVALUATION ENDPOINT ====================

@app.get("/api/stats/daily")
def get_daily_evaluation(
    student_id: str = Query(..., description="Student ID (required)"),
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (default: today)")
) -> Dict:
    """
    Đánh giá học sinh theo ngày dựa trên 3 tiêu chí (cho giáo viên):
    1. Tính tích cực / Mức độ tham gia (0-2 điểm)
    2. Năng lực học tập / Chất lượng làm bài (0-2 điểm)
    3. Tính kỷ luật / Quản lý thời gian (0-1 điểm)
    
    Áp dụng Quality Gating: Chất lượng là tiêu chí quan trọng nhất
    
    Args:
        student_id: Student ID
        date: Date to evaluate (default: today)
        
    Returns:
        Daily evaluation with rating và nhận xét cho giáo viên
    """
    try:
        # Parse date
        if date:
            target_date = date
        else:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get submissions for the day
        conn = submission_manager._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                score,
                duration
            FROM submissions
            WHERE student_id = ? 
            AND DATE(submitted_at) = ?
        """, (student_id, target_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Calculate stats
        total_submissions = len(rows)
        
        if total_submissions == 0:
            return {
                "success": True,
                "student_id": student_id,
                "date": target_date,
                "total_submissions": 0,
                "avg_score": 0.0,
                "on_time_rate": 0.0,
                "participation_score": 0.0,
                "competence_score": 0.0,
                "discipline_score": 0.0,
                "total_score": 0.0,
                "rating": "Yếu",
                "teacher_comment": "Học sinh chưa tham gia làm bài trong ngày này. Cần nhắc nhở."
            }
        
        # Calculate average score
        scores = [row["score"] for row in rows]
        avg_score = sum(scores) / len(scores)
        
        # Calculate on-time rate (duration <= 15 minutes)
        on_time_count = sum(1 for row in rows if row["duration"] <= 15)
        late_count = total_submissions - on_time_count
        on_time_rate = (on_time_count / total_submissions) * 100
        
        # ========== CALCULATE EVALUATION SCORES ==========
        
        # 1. Participation Score - Tính tích cực (0-2 điểm)
        if total_submissions == 0:
            participation_score = 0.0
        elif total_submissions <= 2:
            participation_score = 0.5
        elif total_submissions <= 4:
            participation_score = 1.0
        elif total_submissions <= 7:
            participation_score = 1.5
        else:  # 8+
            participation_score = 2.0
        
        # 2. Competence Score - Năng lực học tập (0-2 điểm)
        if avg_score < 5.0:
            competence_score = 0.0
        elif avg_score < 6.5:
            competence_score = 0.5
        elif avg_score < 7.5:
            competence_score = 1.0
        elif avg_score < 9.0:
            competence_score = 1.5
        else:  # 9.0-10.0
            competence_score = 2.0
        
        # 3. Discipline Score - Tính kỷ luật (0-1 điểm)
        if on_time_rate < 50:
            discipline_score = 0.0
        elif on_time_rate < 70:
            discipline_score = 0.25
        elif on_time_rate < 80:
            discipline_score = 0.5
        elif on_time_rate < 90:
            discipline_score = 0.75
        else:  # 90-100%
            discipline_score = 1.0
        
        # Total score
        total_score = participation_score + competence_score + discipline_score
        
        # ========== QUALITY GATING (OPTION 1) ==========
        # Chất lượng là tiêu chí quan trọng nhất
        
        if avg_score < 5.0:
            # Học sinh yếu về năng lực
            if total_score >= 3.0:
                rating = "Trung bình"
                teacher_comment = f"Học sinh tích cực tham gia ({total_submissions} bài) nhưng năng lực còn hạn chế (điểm TB: {round(avg_score, 1)}). Cần hỗ trợ về phương pháp học tập và nắm vững kiến thức cơ bản."
            else:
                rating = "Yếu"
                teacher_comment = "Học sinh cần được quan tâm và hỗ trợ thêm. Đề xuất liên hệ phụ huynh để tìm hiểu nguyên nhân và có biện pháp hỗ trợ kịp thời."
        
        elif avg_score < 6.5:
            # Học sinh trung bình về năng lực
            if total_score >= 4.0:
                rating = "Khá"
                teacher_comment = "Học sinh tích cực và có năng lực ở mức trung bình khá. Khuyến khích tiếp tục cố gắng để đạt kết quả cao hơn."
            elif total_score >= 3.0:
                rating = "Khá"
                teacher_comment = "Học sinh hoàn thành tốt nhiệm vụ. Cần cố gắng thêm ở chất lượng làm bài để đạt kết quả tốt hơn."
            else:
                rating = "Trung bình"
                teacher_comment = "Học sinh đạt mức cơ bản. Cần tăng cường cả số lượng và chất lượng bài làm."
        
        else:
            # Năng lực tốt (>= 6.5)
            if total_score < 1.5:
                rating = "Trung bình"
                teacher_comment = f"Học sinh có năng lực tốt (điểm TB: {round(avg_score, 1)}) nhưng tham gia rất ít. Cần khuyến khích làm thêm bài để rèn luyện."
            elif total_score < 3.0:
                rating = "Khá"
                teacher_comment = f"Học sinh có năng lực tốt (điểm TB: {round(avg_score, 1)}). Khuyến khích tham gia nhiều hơn để phát triển toàn diện."
            elif total_score < 4.0:
                rating = "Khá"
                teacher_comment = "Học sinh hoàn thành tốt nhiệm vụ học tập. Tiếp tục duy trì và phát huy!"
            elif total_score < 5.0:
                rating = "Giỏi"
                teacher_comment = "Học sinh học tập nghiêm túc và đạt kết quả tốt. Rất đáng khích lệ và khen ngợi!"
            else:  # 5.0
                rating = "Xuất sắc"
                teacher_comment = "Học sinh đạt chuẩn tối ưu về cả tham gia, năng lực và kỷ luật. Xứng đáng được khen thưởng và làm gương cho các bạn khác!"
        
        # ========== SAVE TO DATABASE ==========
        try:
            eval_id = evaluation_storage.save_evaluation({
                "student_id": student_id,
                "date": target_date,
                "total_submissions": total_submissions,
                "avg_score": round(avg_score, 2),
                "on_time_rate": round(on_time_rate, 2),
                "participation_score": participation_score,
                "competence_score": competence_score,
                "discipline_score": discipline_score,
                "total_score": round(total_score, 2),
                "rating": rating,
                "teacher_comment": teacher_comment
            })
            print(f"   💾 Saved to DB: {eval_id}")
        except Exception as e:
            print(f"   ⚠️ Failed to save evaluation: {e}")
            # Don't fail request, just log
        # =======================================
        
        return {
            "success": True,
            "student_id": student_id,
            "date": target_date,
            "total_submissions": total_submissions,
            "avg_score": round(avg_score, 2),
            "on_time_rate": round(on_time_rate, 2),
            "participation_score": participation_score,
            "competence_score": competence_score,
            "discipline_score": discipline_score,
            "total_score": round(total_score, 2),
            "rating": rating,
            "teacher_comment": teacher_comment
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")
    
@app.get("/api/stats/history")
def get_evaluation_history(
    student_id: str = Query(..., description="Student ID (required)"),
    days: int = Query(7, ge=1, le=365, description="Number of recent days (default: 7, max: 365)"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
) -> Dict:
    """
    Get evaluation history from database
    
    Two modes:
    1. Recent days: Get last N days of evaluations
    2. Date range: Get evaluations between start_date and end_date
    
    Args:
        student_id: Student ID
        days: Number of recent days (default: 7)
        start_date: Optional start date
        end_date: Optional end date
        
    Returns:
        List of daily evaluations ordered by date DESC
    """
    try:
        # Get history from database
        history = evaluation_storage.get_history(
            student_id=student_id,
            days=days,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate summary statistics
        if history:
            avg_total_score = sum(h['total_score'] for h in history) / len(history)
            avg_submissions = sum(h['total_submissions'] for h in history) / len(history)
            avg_score = sum(h['avg_score'] for h in history) / len(history)
            
            # Count ratings
            rating_counts = {}
            for h in history:
                rating = h['rating']
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
            
            summary = {
                "total_days": len(history),
                "avg_total_score": round(avg_total_score, 2),
                "avg_daily_submissions": round(avg_submissions, 1),
                "avg_score": round(avg_score, 2),
                "rating_distribution": rating_counts,
                "date_range": {
                    "from": history[-1]['date'] if history else None,
                    "to": history[0]['date'] if history else None
                }
            }
        else:
            summary = {
                "total_days": 0,
                "message": "Chưa có dữ liệu đánh giá"
            }
        
        return {
            "success": True,
            "student_id": student_id,
            "query": {
                "days": days,
                "start_date": start_date,
                "end_date": end_date
            },
            "summary": summary,
            "history": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")
    
@app.post("/api/teacher/comment")
def submit_teacher_comment(
    teacher_id: str = Query(..., description="Teacher ID (required)"),
    teacher_comment: str = Query(..., description="Teacher's comment"),
    teacher_rating: Optional[str] = Query(None, description="Teacher's rating (optional)")
) -> Dict:
    """
    Submit teacher comment with optional rating
    
    Args:
        teacher_id: Teacher ID
        teacher_comment: Comment text
        teacher_rating: Optional rating (e.g., "Xuất sắc", "Giỏi", "Khá", "Trung bình", "Yếu")
        
    Returns:
        Echo back the same data
    """
    from datetime import datetime
    
    response = {
        "success": True,
        "teacher_id": teacher_id,
        "teacher_comment": teacher_comment,
        "submitted_at": datetime.now().isoformat()
    }
    
    # Add rating if provided
    if teacher_rating:
        response["teacher_rating"] = teacher_rating
    
    return response

# ==================== SIMPLE PIPELINE API - GIỐNG MASTER_PIPELINE ====================
# Paste vào cuối app.py

import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict
from fastapi import BackgroundTasks, UploadFile, File, HTTPException

# Import pipeline modules (giống master_pipeline.py)
pipeline_dir = Path(__file__).parent.parent / "pipeline"
sys.path.insert(0, str(pipeline_dir))

from extract_images_from_docx import extract_images_in_order
from extract_text_from_images import process_exam_with_openai
from convert_txt_to_Q import parse_txt_to_json
from convert_Q_to_A import load_json, load_answer_key, map_answers, save_json
from convert_A_to_E import parse_explanations, map_explanations
from build_vector_db import main as build_vector_db_main

# Task tracking
pipeline_tasks = {}

# Fixed config (giống master_pipeline.py)
FIXED_BASE_NAME = "ester-lipid_hoa12"
FIXED_SUBJECT = "Hóa học"

@app.post("/api/pipeline/process-docx")
async def process_docx_to_vectordb(
    background_tasks: BackgroundTasks,
    docx_file: UploadFile = File(...)
) -> Dict:
    """
    Upload DOCX and run pipeline (exact copy of master_pipeline.py logic)
    """
    try:
        if not docx_file.filename.endswith('.docx'):
            raise HTTPException(status_code=400, detail="File must be .docx")
        
        task_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to temp
        temp_dir = Path(tempfile.mkdtemp())
        docx_path = temp_dir / docx_file.filename
        with open(docx_path, 'wb') as f:
            f.write(await docx_file.read())
        
        pipeline_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "filename": docx_file.filename
        }
        
        # Run in background
        background_tasks.add_task(_run_master_pipeline, task_id, str(docx_path))
        
        return {
            "success": True,
            "task_id": task_id,
            "status_url": f"/api/pipeline/status/{task_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_master_pipeline(task_id: str, docx_path: str):
    """
    EXACT COPY of master_pipeline.py logic
    """
    try:
        # Use uploaded filename
        base_name = Path(docx_path).stem
        subject = FIXED_SUBJECT
        
        # Setup paths (giống master_pipeline)
        img_folder = f"data/input/img/{base_name}"
        Path(img_folder).mkdir(parents=True, exist_ok=True)
        
        ocr_output_name = f"ocr_{base_name}"
        json_folder = "data/input/json"
        Path(json_folder).mkdir(parents=True, exist_ok=True)
        
        ocr_txt = f"{img_folder}/{ocr_output_name}.txt"
        json_q = f"{json_folder}/{base_name}_Q.json"
        json_a = f"{json_folder}/{base_name}_A.json"
        json_e = f"{json_folder}/{base_name}_E.json"
        
        # Copy DOCX
        docx_dest = f"data/input/docx/{base_name}.docx"
        Path("data/input/docx").mkdir(parents=True, exist_ok=True)
        shutil.copy(docx_path, docx_dest)
        
        # ========== STEP 1: Extract (GIỐNG MASTER_PIPELINE) ==========
        pipeline_tasks[task_id].update({"status": "running", "progress": 16, "step": "extract"})
        extract_images_in_order(docx_dest, img_folder)
        print(f"✅ Step 1: Extract images")
        
        # ========== STEP 2: OCR (GIỐNG MASTER_PIPELINE) ==========
        pipeline_tasks[task_id].update({"progress": 33, "step": "ocr"})
        result = process_exam_with_openai(
            folder_path=img_folder,
            detail="high",
            model="gpt-4o-mini",
            delay=0.5,
            output_name=ocr_output_name
        )
        print(f"✅ Step 2: OCR")
        
        # ========== STEP 3: Parse (GIỐNG MASTER_PIPELINE) ==========
        pipeline_tasks[task_id].update({"progress": 50, "step": "parse"})
        parse_txt_to_json(ocr_txt, json_q, subject)
        print(f"✅ Step 3: Parse")
        
        # ========== STEP 4: Map Answers (GIỐNG MASTER_PIPELINE) ==========
        pipeline_tasks[task_id].update({"progress": 66, "step": "map_answers"})
        answer_key_txt = f"data/input/txt/{FIXED_BASE_NAME}-A.txt"
        
        if Path(answer_key_txt).exists():
            data = load_json(json_q)
            answer_key = load_answer_key(answer_key_txt)
            mapped_data, missing = map_answers(data, answer_key)
            save_json(mapped_data, json_a)
            print(f"✅ Step 4: Map answers")
        else:
            shutil.copy(json_q, json_a)
            print(f"⏭️  Step 4: Skip (no answer file)")
        
        # ========== STEP 5: Map Explanations (GIỐNG MASTER_PIPELINE) ==========
        pipeline_tasks[task_id].update({"progress": 83, "step": "map_explanations"})
        explanation_txt = f"data/input/txt/{FIXED_BASE_NAME}-E.txt"
        
        if Path(explanation_txt).exists():
            data = load_json(json_a)
            explanation_map = parse_explanations(explanation_txt)
            new_data, mapped, missing = map_explanations(data, explanation_map)
            save_json(new_data, json_e)
            print(f"✅ Step 5: Map explanations")
        else:
            shutil.copy(json_a, json_e)
            print(f"⏭️  Step 5: Skip (no explanation file)")
        
        # ========== STEP 6: Vector DB (GIỐNG MASTER_PIPELINE) ==========
        pipeline_tasks[task_id].update({"progress": 90, "step": "vectordb"})
        
        import build_vector_db
        original_input = build_vector_db.INPUT_JSON
        build_vector_db.INPUT_JSON = json_e
        
        build_vector_db_main()
        
        build_vector_db.INPUT_JSON = original_input
        
        print(f"✅ Step 6: Build vector DB")
        
        # ========== DONE ==========
        pipeline_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "step": "done",
            "completed_at": datetime.now().isoformat(),
            "output_files": {
                "questions": json_q,
                "final": json_e
            }
        })
        
        print(f"🎉 Pipeline completed: {task_id}")
        
    except Exception as e:
        pipeline_tasks[task_id].update({
            "status": "failed",
            "error": str(e)
        })
        print(f"❌ Pipeline failed: {e}")


@app.get("/api/pipeline/status/{task_id}")
def get_pipeline_status(task_id: str) -> Dict:
    """Get status"""
    if task_id not in pipeline_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {"success": True, "task_id": task_id, **pipeline_tasks[task_id]}


@app.get("/api/pipeline/tasks")
def list_pipeline_tasks() -> Dict:
    """List tasks"""
    tasks = [{"task_id": tid, **data} for tid, data in pipeline_tasks.items()]
    return {"success": True, "tasks": tasks}


# ========== ENDPOINT ==========

class MarkdownToPDFRequest(BaseModel):
    markdown_content: str = Field(..., description="Markdown content")
    filename: Optional[str] = Field("output.pdf", description="Output filename")
    toc: Optional[bool] = Field(False, description="Include table of contents")
    number_sections: Optional[bool] = Field(False, description="Number sections")


# ========== CONVERT ENDPOINT ==========
@app.post("/api/markdown/to-pdf")
async def convert_markdown_to_pdf(request: MarkdownToPDFRequest):
    """
    Convert Markdown to PDF
    
    **Returns:** JSON with download link
```json
    {
        "success": true,
        "file_id": "abc123",
        "download_url": "http://localhost:8110/api/download-pdf/abc123",
        "filename": "output.pdf",
        "expires_in": 3600
    }
```
    """
    try:
        print(f"📝 Converting markdown to PDF...")
        
        # Cleanup old files
        cleanup_old_pdfs()
        
        # Validate filename
        filename = request.filename
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Output to storage directory
        output_path = PDF_STORAGE_DIR / f"{file_id}.pdf"
        
        # Preprocess markdown
        markdown_content = request.markdown_content
        if True:  # clean_markdown
            markdown_content = preprocess_markdown(markdown_content)
        
        # Create temp markdown file
        temp_md = PDF_STORAGE_DIR / f"{file_id}.md"
        with open(temp_md, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Build pandoc command
        cmd = [
            'pandoc',
            str(temp_md),
            '-o', str(output_path),
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=12pt',
            '-V', 'documentclass=article',
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=blue',
            '-V', 'urlcolor=blue',
            '--highlight-style=tango',
            '--standalone'
        ]
        
        if request.toc:
            cmd.extend(['--toc', '--toc-depth=3'])
        
        if request.number_sections:
            cmd.append('--number-sections')
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Clean up temp markdown
        temp_md.unlink()
        
        # Check if PDF was created
        if not output_path.exists():
            raise RuntimeError("PDF was not created")
        
        # Store in memory
        PDF_STORAGE[file_id] = (str(output_path), time.time())
        
        print(f"✅ PDF created: {output_path}")
        
        # Build download URL
        base_url = os.getenv("API_BASE_URL", "http://14.225.211.7:8110")
        download_url = f"{base_url}/api/download-pdf/{file_id}"
        
        return {
            "success": True,
            "file_id": file_id,
            "download_url": download_url,
            "filename": filename,
            "expires_in": 3600,  # 1 hour
            "message": "PDF created successfully. Download link expires in 1 hour."
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pandoc error: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Pandoc error: {e.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== DOWNLOAD ENDPOINT ==========
@app.get("/api/download-pdf/{file_id}")
async def download_pdf(file_id: str):
    """
    Download PDF by file ID
    
    **Returns:** PDF file
    """
    try:
        # Check if file exists
        if file_id not in PDF_STORAGE:
            raise HTTPException(status_code=404, detail="File not found or expired")
        
        pdf_path, created_time = PDF_STORAGE[file_id]
        
        # Check if expired (1 hour)
        if time.time() - created_time > 3600:
            # Clean up
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            del PDF_STORAGE[file_id]
            raise HTTPException(status_code=410, detail="File has expired")
        
        # Check if file exists on disk
        if not os.path.exists(pdf_path):
            del PDF_STORAGE[file_id]
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Get original filename from request (default to file_id)
        filename = f"{file_id}.pdf"
        
        print(f"📥 Downloading: {pdf_path}")
        
        return FileResponse(
            path=pdf_path,
            media_type='application/pdf',
            filename=filename,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ==================== END ====================

# ==================== RUN INFO ====================
if __name__ == "__main__":
    import uvicorn
    print("⚠️  Don't run this file directly!")
    print("👉 Use: python run_api.py")