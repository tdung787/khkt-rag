"""
tools/quiz_generator.py

Tool để tạo đề kiểm tra trắc nghiệm bằng AI
Cost: ~$0.015/quiz (10 câu) - optimized
"""

import re
import os
import json
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI 


def load_student_profile(
    student_id: str, 
    api_base_url: str = None
) -> Optional[Dict]:
    """
    Load student profile from DATABASE (avoid deadlock)
    
    Args:
        student_id: Student ID
        api_base_url: Not used anymore
        
    Returns:
        Student profile with difficulty_level from database
    """
    import sqlite3
    from datetime import datetime
    
    try:
        print(f"📥 Loading student profile from database...")
        print(f"   Student ID: {student_id}")
        
        # Connect to correct database
        db_path = "database/student_evaluations.db"  # ← TÊN ĐÚNG
        
        if not os.path.exists(db_path):
            print(f"⚠️  Database not found, using default difficulty")
            return {
                "_id": student_id,
                "difficulty_level": "medium",
                "error": "Database not found"
            }
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get latest evaluation for student
        cursor.execute("""
            SELECT * FROM daily_evaluations
            WHERE student_id = ?
            ORDER BY date DESC
            LIMIT 1
        """, (student_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print(f"⚠️  No evaluation found, using default difficulty")
            return {
                "_id": student_id,
                "difficulty_level": "medium",
                "error": "No evaluation data"
            }
        
        eval_data = dict(row)
        rating = eval_data.get("rating", "")
        avg_score = eval_data.get("avg_score", 0.0)
        total_score = eval_data.get("total_score", 0.0)
        
        print(f"   📊 Rating: {rating}")
        print(f"   📈 Avg Score: {avg_score}")
        print(f"   🎯 Total Score: {total_score}")
        
        # Map rating to difficulty level
        if "Xuất sắc" in rating:
            difficulty_level = "hard"
            recommendation = "Học sinh xuất sắc - Đề khó để thử thách"
        elif "Giỏi" in rating:
            difficulty_level = "hard"
            recommendation = "Học sinh giỏi - Đề khó"
        elif "Khá" in rating:
            difficulty_level = "medium"
            recommendation = "Học sinh khá - Đề trung bình-khó"
        elif "Trung bình" in rating:
            difficulty_level = "medium"
            recommendation = "Học sinh trung bình - Đề trung bình"
        elif "Yếu" in rating:
            difficulty_level = "easy"
            recommendation = "Học sinh cần hỗ trợ - Đề dễ"
        else:
            difficulty_level = "medium"
            recommendation = "Chưa có đánh giá - Đề trung bình"
        
        print(f"   🎓 Difficulty Level: {difficulty_level}")
        print(f"   💡 {recommendation}")
        
        # Build profile
        profile = {
            "_id": student_id,
            "difficulty_level": difficulty_level,
            "rating": rating,
            "avg_score": avg_score,
            "total_score": total_score,
            "recommendation": recommendation,
            "evaluated_date": eval_data.get("date")
        }
        
        print(f"✅ Student profile loaded from database")
        
        return profile
        
    except Exception as e:
        print(f"⚠️  Error loading profile: {e}")
        return {
            "_id": student_id,
            "difficulty_level": "medium",
            "error": str(e)
        }

def get_difficulty_vietnamese(difficulty_pref: str) -> str:
    """Convert difficulty to Vietnamese"""
    mapping = {"easy": "dễ", "medium": "trung bình", "hard": "khó"}
    return mapping.get(difficulty_pref.lower(), "trung bình")


class QuizGenerator:
    """Generate quiz using AI"""
    
    def __init__(
        self, 
        client: OpenAI, 
        student_id: str = None, 
        api_base_url: str = None
    ):
        self.client = client
        self.student_id = student_id
        
        if api_base_url is None:
            api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8110')
        
        self.api_base_url = api_base_url
        
        # ========== FIX: LAZY LOADING - KHÔNG GỌI NGAY ==========
        self.student_profile = None
        
        if student_id:
            print(f"   📋 QuizGenerator initialized for student: {student_id}")
            print(f"   💡 Profile will be loaded when generating quiz")
        else:
            print("⚠️  No student_id provided")
        # ========================================================
    
    def _ensure_profile_loaded(self):
        """Lazy load profile if not already loaded"""
        if self.student_id and not self.student_profile:
            print(f"   📥 Loading profile for {self.student_id}...")
            self.student_profile = load_student_profile(
                self.student_id, 
                self.api_base_url
            )
            
            if self.student_profile:
                diff = self.student_profile.get("difficulty_level", "medium")
                rating = self.student_profile.get("rating", "N/A")
                print(f"   ✓ Profile loaded: {rating} → Độ khó: {get_difficulty_vietnamese(diff)}")
    
    def get_student_info(self) -> Dict:
        """Get formatted student info"""
        # Ensure profile is loaded
        self._ensure_profile_loaded()
        
        if not self.student_profile:
            return {
                "full_name": "........................",
                "current_class": "........................",
                "difficulty": "trung bình",
                "grade_level": None
            }
        
        user_info = self.student_profile.get("user_id", {})
        return {
            "full_name": user_info.get("full_name", "........................"),
            "current_class": self.student_profile.get("current_class", "........................"),
            "difficulty": get_difficulty_vietnamese(self.student_profile.get("difficulty_level", "medium")),
            "grade_level": self.student_profile.get("grade_level")
        }
    
    def generate_quiz(
        self,
        subject: str,
        topic: str,
        difficulty: str = None,
        use_student_difficulty: bool = True
    ) -> Dict:
        """Generate quiz - Fixed for 15-min, 10 questions"""
        
        # ========== LAZY LOAD PROFILE HERE ==========
        self._ensure_profile_loaded()
        # ============================================
        
        # Force 15-min, 10 questions format
        num_questions = 10
        time_limit = 15
        
        student_info = self.get_student_info()

        # Use student difficulty preference
        if use_student_difficulty or difficulty is None:
            difficulty = student_info["difficulty"]
        else:
            if difficulty.lower() in ["easy", "medium", "hard"]:
                difficulty = get_difficulty_vietnamese(difficulty)
        
        print(f"\n📝 Tạo đề: {subject} - {topic}")
        print(f"   👤 {student_info['full_name']} ({student_info['current_class']})")
        print(f"   📊 10 câu - 15 phút - Độ khó: {difficulty}")
        
        # Optimized system prompt (reduced tokens)
        system_prompt = """Chuyên gia ra đề trắc nghiệm THPT. Tạo đề 15 phút, 10 câu.

QUY TẮC:
1. BẮT BUỘC: Đúng 10 câu (Câu 1→10)
2. Câu hỏi chính xác khoa học
3. Đáp án nhiễu hợp lý
4. CHỈ ĐỀ, KHÔNG ĐÁP ÁN TRONG NỘI DUNG

ĐỘ KHÓ (15 phút):
- Dễ: Nhớ định nghĩa, 1 bước tính, số đẹp. VD: "v=s/t với s=100m, t=10s"
- TB: 2-3 bước, so sánh khái niệm. VD: "v trung bình khi v đổi"
- Khó: 3-4 bước, kết hợp 2-3 công thức, bẫy nhỏ. VD: "đi-về khác v, tính s"

FORMAT ĐỀ BÀI:
# ĐỀ KIỂM TRA 15 PHÚT - [MÔN]
**Chủ đề**: [topic]
**Độ khó**: [level]
**Thời gian**: 15 phút
**Tổng điểm**: 10 điểm

---

## **Câu 1**: [question]
**A.** [option]  
**B.** [option]  
**C.** [option]  
**D.** [option]

## **Câu 2**: [question]
**A.** [option]  
**B.** [option]  
**C.** [option]  
**D.** [option]

[... Câu 3-9 ...]

## **Câu 10**: [question]
**A.** [option]  
**B.** [option]  
**C.** [option]  
**D.** [option]

---
_Hết_

⚠️ QUAN TRỌNG: SAU KHI TẠO ĐỀ XONG, THÊM DÒNG ẨN Ở CUỐI:
<!-- ANSWER_KEY: 1-X,2-Y,3-Z,4-W,5-V,6-U,7-T,8-S,9-R,10-Q -->

Trong đó X,Y,Z,... là đáp án đúng (A/B/C/D) của từng câu.
VÍ DỤ: Nếu câu 1 đáp án A, câu 2 đáp án B → <!-- ANSWER_KEY: 1-A,2-B,... -->
"""
        
        # Optimized user prompt (reduced tokens)
        difficulty_extra = ""
        if difficulty == "khó":
            difficulty_extra = "\n⚠️ Độ khó 'khó': 6-7 câu bài tập 3-4 bước, đáp án gần nhau, tối đa 2-3 câu lý thuyết."
        
        user_prompt = f"""Đề thi:
- Môn: {subject} | Chủ đề: {topic}
- Học sinh: {student_info['full_name']} - {student_info['current_class']}
- 10 câu, 15 phút, mỗi câu 1 điểm
- Độ khó: {difficulty}{difficulty_extra}

Yêu cầu: Đúng 10 câu, 4 đáp án/câu, không đáp án. Tập trung chủ đề "{topic}"."""
        
        try:
            print("   🤖 Đang sinh đề...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            quiz_markdown = response.choices[0].message.content.strip()
            
            answer_key = self._extract_answer_key(quiz_markdown)
            
            if not answer_key:
                print("   ⚠️ Không tìm thấy answer key, đang retry...")
                # Retry with emphasis
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\n⚠️ CHÚ Ý: BẮT BUỘC phải thêm dòng:\n<!-- ANSWER_KEY: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B -->"}
                    ],
                    temperature=0.7,
                    max_tokens=3000
                )
                quiz_markdown = response.choices[0].message.content.strip()
                answer_key = self._extract_answer_key(quiz_markdown)
            
            # Validate
            if not self._validate_quiz(quiz_markdown, num_questions):
                print("   ⚠️ Retry...")
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\nCHÚ Ý: Đúng format ## **Câu X**: và **A.**, **B.**, **C.**, **D.**"}
                    ],
                    temperature=0.7,
                    max_tokens=3000
                )
                quiz_markdown = response.choices[0].message.content.strip()
            
            print("   ✓ Hoàn thành!")
            
            metadata = self._extract_metadata(quiz_markdown)
            
            return {
                "success": True,
                "quiz_markdown": quiz_markdown,
                "answer_key": answer_key,
                "metadata": {
                    "subject": subject,
                    "topic": topic,
                    "num_questions": num_questions,
                    "difficulty": difficulty,
                    "time_limit": time_limit,
                    "student_info": student_info,
                    **metadata
                }
            }
            
        except Exception as e:
            print(f"   ✗ Lỗi: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_quiz(self, quiz_markdown: str, expected_questions: int) -> bool:
        """Validate quiz format"""
        question_pattern = r'##\s+\*\*Câu\s+\d+\*\*:'
        questions = re.findall(question_pattern, quiz_markdown)
        
        if len(questions) != expected_questions:
            print(f"   ⚠️ Số câu: {len(questions)}/{expected_questions}")
            return False
        
        option_pattern = r'\*\*[A-D]\.\*\*'
        options = re.findall(option_pattern, quiz_markdown)
        
        if len(options) != expected_questions * 4:
            print(f"   ⚠️ Số đáp án: {len(options)}/{expected_questions * 4}")
            return False
        
        if "ĐÁP ÁN" in quiz_markdown.upper():
            print("   ⚠️ Có đáp án")
            return False
        
        return True
    
    def _extract_metadata(self, quiz_markdown: str) -> Dict:
        """Extract metadata"""
        question_pattern = r'##\s+\*\*Câu\s+\d+\*\*:'
        questions = re.findall(question_pattern, quiz_markdown)
        return {"total_questions_found": len(questions)}
    
       
    def _extract_answer_key(self, markdown: str) -> Optional[str]:
        """Extract answer key từ markdown"""
        try:
            # ========== PATTERN 1: HTML Comment ==========
            # <!-- ANSWER_KEY: 1-A,2-B,3-C,... -->
            pattern1 = r'<!--\s*ANSWER_KEY:\s*([0-9]+-[A-D](?:,\s*[0-9]+-[A-D])+)\s*-->'
            match = re.search(pattern1, markdown, re.IGNORECASE)
            if match:
                return match.group(1).replace(" ", "")
            
            # ========== PATTERN 2: Inline format ==========
            # "Đáp án: 1-A,2-B,3-C,..."
            pattern2 = r'(?:Đáp án:|Answer key:|Answers:)\s*([0-9]+-[A-D](?:,\s*[0-9]+-[A-D])+)'
            match = re.search(pattern2, markdown, re.IGNORECASE)
            if match:
                return match.group(1).replace(" ", "")
            
            # ========== PATTERN 3: List format ==========
            # **Đáp án:**
            # 1. D
            # 2. A
            # 3. B
            pattern3 = r'\*\*Đáp án:\*\*\s*\n((?:\d+\.\s*[A-D]\s*\n?)+)'
            match = re.search(pattern3, markdown, re.IGNORECASE | re.MULTILINE)
            if match:
                lines = match.group(1).strip().split('\n')
                answers = []
                for line in lines:
                    # Extract "1. D" → "1-D"
                    m = re.match(r'(\d+)\.\s*([A-D])', line.strip())
                    if m:
                        answers.append(f"{m.group(1)}-{m.group(2)}")
                
                if len(answers) == 10:
                    return ",".join(answers)
            
            # ========== PATTERN 4: Anywhere in text ==========
            # Tìm 10 dòng liên tiếp có format "1. X"
            lines = markdown.split('\n')
            answers = []
            for line in lines:
                m = re.match(r'^\s*(\d+)\.\s*([A-D])\s*$', line.strip())
                if m:
                    answers.append(f"{m.group(1)}-{m.group(2)}")
                    if len(answers) == 10:
                        return ",".join(answers)
                elif answers:  # Reset nếu gặp dòng không match
                    answers = []
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Extract answer key error: {e}")
            return None

def extract_topic_from_query(query: str, openai_client: OpenAI) -> Optional[Dict]:
    """Extract subject and topic from query"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Trích xuất từ: "{query}"

JSON:
{{
    "subject": "Tên môn học",
    "topic": "chủ đề CỤ THỂ",
    "user_difficulty": "dễ"|"trung bình"|"khó"|null
}}

QUY TẮC:
- LUÔN trả về tên môn học (Toán, Vật lý, Hóa học, Sinh học, Văn, Anh, Sử, Địa, Tin, ...)
- KHÔNG bao giờ trả về subject = null
- Nếu chủ đề chung chung → Cụ thể hóa
- Chỉ set "user_difficulty" khi user NÓI RÕ

VD:
"Đề 15p Động lực học độ khó TB" → {{"subject":"Vật lý","topic":"Ba định luật Newton","user_difficulty":"trung bình"}}
"Tạo đề Văn về Chiếc lược ngà" → {{"subject":"Văn","topic":"Chiếc lược ngà","user_difficulty":null}}
"Tạo đề Lịch sử thế giới" → {{"subject":"Lịch sử","topic":"Lịch sử thế giới","user_difficulty":null}}
"15 câu Toán khó Hệ BPT" → {{"subject":"Toán","topic":"Hệ bất phương trình bậc nhất hai ẩn","user_difficulty":"khó"}}
"Cho tôi bài kiểm tra Tiếng Anh" → {{"subject":"Tiếng Anh","topic":"Grammar","user_difficulty":null}}

Chỉ JSON."""
            }],
            temperature=0,
            max_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # ========== SỬA VALIDATION ==========
        # Trước: if result.get("subject") and result.get("topic"):
        # Sau: Chỉ check topic, không check subject trong allowed list
        if result.get("topic"):
            return result
        
        return None
        # ====================================
        
    except Exception as e:
        print(f"⚠️ Extract error: {e}")
        return None