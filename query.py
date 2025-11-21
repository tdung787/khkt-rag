import os
import json
import re
import platform
import subprocess
import requests
import threading
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import graph tool and quiz tool
from src.tools.graph_generator import (
    GraphGenerator,
    extract_equation_from_query,
    extract_range_from_query
)
from src.tools.quiz_generator import (
    QuizGenerator,
    extract_topic_from_query
)
from src.tools.quiz_storage import QuizStorage
from src.tools.quiz_guard import QuizGuard
from src.tools.submission_manager import SubmissionManager

load_dotenv()

# ================== CONFIG ==================
OPENAI_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
QDRANT_PATH = "database/qdrant_storage"
COLLECTION_NAME = "KHTN_QA"

# Supported subjects
SUBJECTS = {
    "Vật lý": ["vật lý", "physics", "lực", "năng lượng", "điện", "từ", "quang", "nhiệt"],
    "Hóa học": ["hóa học", "chemistry", "phản ứng", "nguyên tố", "hợp chất", "ion"],
    "Sinh học": [
        "gen", "adn", "arn", "protein", "tế bào", "NST", "nhiễm sắc thể",
        "đột biến", "nucleotit", "adenin", "guanin", "timin", "xitozin",
        "liên kết hidro", "giảm phân", "nguyên phân", "kiểu gen", "kiểu hình",
        "di truyền", "alen", "dna", "rna", "enzyme", "hạt phấn"
    ],
    "Toán": ["toán", "math", "phương trình", "hàm số", "đồ thị", "số học"]
}
# Allowed subjects for quiz generation
ALLOWED_QUIZ_SUBJECTS = ["Toán", "Vật lý", "Hóa học", "Sinh học"]

# ================== INTENT CLASSIFIER ==================
class IntentClassifier:
    """Classify user query intent using LLM"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def classify(self, query: str) -> Dict:
        """Classify query intent"""
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là trợ lý phân loại câu hỏi học sinh.

Phân tích câu hỏi và xác định:
1. Có phải câu hỏi về môn học tự nhiên không? (Toán, Lý, Hóa, Sinh)
2. Nếu có, thuộc môn nào?

Trả về JSON với format:
{
    "is_subject_question": true/false,
    "subject": "Vật lý" | "Hóa học" | "Sinh học" | "Toán" | null,
    "confidence": 0.0-1.0,
    "reasoning": "lý do ngắn gọn"
}

Ví dụ:
- "Định luật Newton là gì?" → {"is_subject_question": true, "subject": "Vật lý", "confidence": 0.95, "reasoning": "Câu hỏi về định luật vật lý"}
- "Hôm nay thời tiết thế nào?" → {"is_subject_question": false, "subject": null, "confidence": 0.9, "reasoning": "Không liên quan môn học"}
"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0
            )
            
            # Parse JSON from response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            return {
                "is_subject_question": result.get("is_subject_question", False),
                "subject": result.get("subject"),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            print(f"⚠️  Lỗi classify: {e}")
            return {
                "is_subject_question": False,
                "subject": None,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }

# ================== RETRIEVAL TOOL ==================
class QuestionRetriever:
    """Retrieve relevant questions from Qdrant"""
    
    def __init__(self, client: OpenAI, qdrant_path: str, collection_name: str):
        self.openai_client = client
        self.qdrant_client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
    
    def _embed_text(self, text: str) -> List[float]:
        """Embed text using OpenAI"""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def _clean_query(self, query: str) -> str:
        """
        Remove multiple choice options from query
        
        Example:
        Input: "Question text\nA. Option A\nB. Option B\nC. Option C\nD. Option D"
        Output: "Question text"
        """
        # Remove options pattern: "A. text" or "A: text" until next option or end
        pattern = r'\n?[A-D][\.:]\s*.+?(?=\n[A-D][\.:]|\Z)'
        cleaned = re.sub(pattern, '', query, flags=re.DOTALL | re.MULTILINE)
        
        # Clean up whitespace
        cleaned = re.sub(r'\n{2,}', '\n', cleaned).strip()
        
        return cleaned
    
    def search(
        self, 
        query: str, 
        subject: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """Search for relevant questions"""
        try:
            # ========== CLEAN QUERY ==========
            # clean_query = self._clean_query(query)
            
            # if len(clean_query) != len(query):
            #     print(f"   🧹 Removed options: {len(query)} → {len(clean_query)} chars")
            # =================================
            
            # Embed query
            query_vector = self._embed_text(query)
            
            # Build filter if subject specified
            search_filter = None
            if subject:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="subject",
                            match=MatchValue(value=subject)
                        )
                    ]
                )
            
            # Search
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            )
            
            # Format results (giữ nguyên)
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "question": result.payload.get("question", ""),
                    "options": result.payload.get("options", {}),
                    "correct_answer": result.payload.get("correct_answer", ""),
                    "correct_answer_text": result.payload.get("correct_answer_text", ""),
                    "question_id": result.payload.get("id", ""),
                    "primary_page": result.payload.get("primary_page", ""),
                    "subject": result.payload.get("subject", ""),
                    "explanation": result.payload.get("explanation", ""),
                    "score": result.score
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️  Lỗi search: {e}")
            return []

# ================== TOOL FUNCTION ==================
def search_questions_tool(
    query: str, 
    intent_classifier: IntentClassifier,
    retriever: QuestionRetriever
) -> str:
    """
    Tool function to search questions
    Args:
        query: User query
        intent_classifier: Intent classifier instance
        retriever: Question retriever instance
    Returns:
        Formatted search results
    """
    # Classify intent
    intent = intent_classifier.classify(query)
    print(f"\n🔍 Intent Classification:")
    print(f"   - Is subject question: {intent['is_subject_question']}")
    print(f"   - Subject: {intent['subject']}")
    print(f"   - Confidence: {intent['confidence']:.2f}")
    print(f"   - Reasoning: {intent['reasoning']}")
    
    if not intent['is_subject_question'] or intent['confidence'] < 0.7:
        return "Câu hỏi này không liên quan đến môn học tự nhiên. Tôi không thể tìm kiếm trong database."
    
    # Search with subject filter
    results = retriever.search(
        query=query,
        subject=intent['subject'],
        top_k=3
    )
    
    if not results:
        return f"Không tìm thấy câu hỏi liên quan về {intent['subject']}."
    
    # Format results
    output = f"Tìm thấy {len(results)} câu hỏi liên quan:\n\n"
    
    for i, result in enumerate(results, 1):
        output += f"--- Câu hỏi {i} (Độ tương đồng: {result['score']:.2f}) ---\n"
        output += f"ID: {result['question_id']}\n"
        output += f"Môn: {result['subject']}\n"
        output += f"Câu hỏi: {result['question']}\n"
        output += f"Các lựa chọn:\n"
        
        for key, value in result['options'].items():
            marker = "✓" if key == result['correct_answer'] else " "
            output += f"  [{marker}] {key}. {value}\n"
        
        output += f"Đáp án đúng: {result['correct_answer']} - {result['correct_answer_text']}\n"
        
        # Thêm explanation nếu có
        if result.get('explanation'):
            output += f"Giải thích: {result['explanation']}\n"
        
        output += "\n"
    
    return output

# ================== SIMPLE AGENT (without LangChain) ==================
class SimpleAgent:
    """Simple agent implementation without LangChain"""
    
    def __init__(self, client: OpenAI, intent_classifier: IntentClassifier, retriever: QuestionRetriever, student_id: str = None):
        self.client = client
        self.intent_classifier = intent_classifier
        self.retriever = retriever
        self.student_id = student_id
        self.graph_generator = GraphGenerator(client)
        self.quiz_generator = QuizGenerator(client, student_id=student_id)
        self.quiz_storage = QuizStorage()
        self.quiz_guard = QuizGuard(client)
        self.submission_manager = SubmissionManager()
        self.conversation_history = []
    
    def _get_system_prompt(self, mode: str = "general") -> str:
        """Get system prompt with real-time pending quiz check"""
        
        # Get student profile
        student_info = ""
        student_id = "unknown"
        if self.quiz_generator.student_profile:
            profile = self.quiz_generator.student_profile
            student_id = profile.get('_id', 'unknown')
            student_info = f"""
    THÔNG TIN HỌC SINH:
    - Họ tên: {profile.get('name', 'N/A')}
    - Lớp: {profile.get('grade', 'N/A')}
    - Độ khó phù hợp: {profile.get('difficulty_level', 'N/A')}
    """
        
        # Check pending quiz
        pending_quiz = self.quiz_storage.get_latest_pending_quiz(student_id)
        pending_warning = ""
        if pending_quiz:
            pending_warning = f"""
    ⚠️⚠️⚠️ CẢNH BÁO QUAN TRỌNG ⚠️⚠️⚠️
    HỌC SINH ĐANG CÓ BÀI KIỂM TRA CHƯA NỘP!
    - Quiz ID: {pending_quiz['id']}
    - Môn: {pending_quiz.get('subject', 'N/A')}
    - Chủ đề: {pending_quiz.get('topic', 'N/A')}
    """
        
        # Build prompt based on mode
        if mode == "search":
            prompt = f"""Bạn là trợ lý giáo dục thông minh.

{student_info}
{pending_warning}

NHIỆM VỤ:
- Trả lời câu hỏi dựa trên ĐÁP ÁN và GIẢI THÍCH được cung cấp

QUY TẮC:
- Giữ nguyên: con số, công thức, ký hiệu
- Giải thích TẠI SAO đáp án đúng
- Ngắn gọn, dễ hiểu

⚠️ ĐỊNH DẠNG BẮT BUỘC (KHÔNG SAI):
**Đáp án [Copy chính xác nội dung đáp án từ "ĐÁP ÁN:" ở dưới]**

**Giải thích:**
Trích nguyên văn, không thêm bớt.

"""
            print(f"   📝 System prompt (search): {len(prompt)} chars")
            return prompt
        
        # Default: general mode
        prompt = f"""Bạn là trợ lý học tập AI dành cho học sinh THPT Việt Nam, hỗ trợ các môn khoa học tự nhiên (Toán, Lý, Hóa, Sinh).

VAI TRÒ:
- Giải thích kiến thức và hướng dẫn tư duy cho 4 môn tự nhiên.

NGỮ CẢNH:
- Nếu học sinh hỏi "hình ảnh vừa nãy", "câu hỏi vừa rồi", "bài trước" → Tìm trong lịch sử trò chuyện message có prefix [📸 Từ ảnh]
- Trả về NỘI DUNG của message đó (phần sau prefix)

PHONG CÁCH:
- Thân thiện, dễ hiểu.
- Có ví dụ khi cần; luôn tích cực.

PHẠM VI (BẮT BUỘC):
✘ Không hỗ trợ gian lận hoặc giải bài kiểm tra đang làm.

KHI NHẬN CÂU HỎI NGOÀI PHẠM VI:
- Lịch sự từ chối.
- Nhắc lại phạm vi 4 môn tự nhiên.
- Gợi ý đặt câu hỏi phù hợp.

MỤC TIÊU:
Hỗ trợ học sinh phát triển tư duy khoa học và kỹ năng tự học bền vững.

"""
        print(f"   📝 System prompt (general): {len(prompt)} chars")
        return prompt
    
    # ==================================================
        
    def _should_use_tool(self, query: str) -> bool:
        """Decide if should search database"""
        
        # ========== PRIORITY 1: MULTIPLE CHOICE ==========
        # Normalize query
        query_normalized = query.upper().replace(" ", "").replace("\n", "")
        
        has_options = (
            "A." in query_normalized and 
            "B." in query_normalized and
            "C." in query_normalized
        )
        
        if has_options:
            print("   ✓ Has options → SEARCH")
            return True
        
        # ========== PRIORITY 2: BLACKLIST KEYWORDS ==========
        # Chặn những từ khóa CHẮC CHẮN không phải 4 môn
        blacklist = [
            # Lịch sử
            "bác hồ", "hồ chí minh", "lịch sử", "chiến tranh", "cách mạng",
            "năm nào", "thế kỷ", "triều đại", "vua", "hoàng đế", "nhà",
            "cổ đại", "trung đại", "cận đại", "hiện đại", "phong kiến",
            "độc lập", "giải phóng", "thống nhất", "đế quốc", "thuộc địa",
            
            # Văn học
            "văn học", "thơ", "ca dao", "tục ngữ", "truyện", "tiểu thuyết",
            "tác giả", "tác phẩm", "nhà văn", "nhà thơ", "chữ hán",
            "truyền kỳ", "ngôn tình", "cổ tích", "thần thoại", "truyền thuyết",
            "văn xuôi", "văn vần", "luận điểm", "nghệ thuật", "tu từ",
            "chiếc lược ngà", "vợ chồng a phủ", "chí phèo", "lão hạc",
            
            # Địa lý
            "địa lý", "địa hình", "khí hậu", "nhiệt đới", "ôn đới",
            "châu lục", "lục địa", "đại dương", "biển", "sông", "núi",
            "đồng bằng", "cao nguyên", "thủ đô", "tỉnh", "thành phố",
            "dân số", "dân cư", "di cư", "kinh tế", "nông nghiệp",
            "công nghiệp", "thương mại", "du lịch", "giao thông",
            
            # Tiếng Anh
            "tiếng anh", "english", "grammar", "vocabulary", "tense",
            "present", "past", "future", "perfect", "continuous",
            "reading", "listening", "speaking", "writing",
            "pronunciation", "accent", "idiom", "phrasal verb",
            
            # Tin học
            "tin học", "máy tính", "computer", "code", "lập trình",
            "python", "java", "javascript", "c++", "html", "css",
            "database", "sql", "algorithm", "data structure",
            "software", "hardware", "network", "internet", "website",
            
            # Giáo dục công dân / GDCD
            "công dân", "gdcd", "pháp luật", "hiến pháp", "quyền",
            "nghĩa vụ", "dân chủ", "nhân quyền", "đạo đức", "lương tâm",
            "trách nhiệm", "xã hội", "cộng đồng", "văn hóa", "truyền thống",
            
            # Thể dục / Âm nhạc / Mỹ thuật
            "thể dục", "thể thao", "bóng đá", "bóng rổ", "chạy", "nhảy",
            "âm nhạc", "nhạc", "ca hát", "nhạc cụ", "giai điệu",
            "mỹ thuật", "vẽ", "tranh", "điêu khắc", "kiến trúc",
            
            # Tôn giáo / Triết học
            "phật giáo", "thiên chúa giáo", "hồi giáo", "đạo",
            "triết học", "triết lý", "tư tưởng", "chủ nghĩa",
            
            # Chính trị / Xã hội
            "đảng", "chính phủ", "quốc hội", "tổng thống", "thủ tướng",
            "bầu cử", "dân chủ", "độc tài", "xã hội chủ nghĩa",
            
            # Kinh tế thực tế (không phải môn học)
            "giá cả", "thị trường", "chứng khoán", "bất động sản",
            "lạm phát", "tỷ giá", "ngân hàng", "tiền tệ",
            
            # Thời sự / Đời sống
            "tin tức", "thời sự", "báo chí", "truyền thông",
            "covid", "dịch bệnh", "bệnh viện", "bác sĩ", "y tế",
            "bóng đá việt nam", "world cup", "olympic"
        ]
        
        query_lower = query.lower()
    
        for keyword in blacklist:
            if keyword in query_lower:
                print(f"   ✗ Blacklist keyword '{keyword}' → SKIP")
                return False
        
        # ========== PRIORITY 3: TRY SEARCH ==========
        print("   ? Ambiguous → TRY SEARCH (will check score)")
        return True

    def _should_use_tool_fallback(self, query: str) -> bool:
        """Deprecated - no longer used"""
        return True
    
    def _should_draw_graph(self, query: str) -> bool:
        """Detect if query asks for graph"""
        graph_keywords = ["vẽ đồ thị", "vẽ đồ", "đồ thị", "graph", "plot", "vẽ hàm"]
        return any(kw in query.lower() for kw in graph_keywords)
    
    def _should_create_quiz(self, user_query: str) -> bool:
        """
        Detect quiz creation intent
        
        Uses hybrid approach:
        1. Keyword matching (primary - fast & reliable)
        2. Regex patterns (backup - catch edge cases)
        
        Returns:
            True if user wants to create a quiz
        """
        query_lower = user_query.lower()
        
        # ========== METHOD 1: KEYWORD MATCHING ==========
        # Simple, fast, covers 95% of cases
        quiz_keywords = [
            # Core keywords
            "tạo đề", "ra đề", "đề thi", "bài kiểm tra",
            
            # English
            "quiz", "test",
            
            # Variants
            "trắc nghiệm", "15 phút", "30 phút",
            
            # Short forms
            "kiểm tra", "bài thi",
            
            # Request patterns
            "cho tôi bài", "cho em bài", "cho mình bài",
            "cho tôi đề", "cho em đề", "cho mình đề",
            
            # Action verbs
            "tạo bài", "ra bài", "làm bài",
            "muốn bài", "cần bài", "muốn đề", "cần đề"
        ]
        
        for keyword in quiz_keywords:
            if keyword in query_lower:
                print(f"   ✓ Matched keyword: '{keyword}'")
                return True
        
        # ========== METHOD 2: REGEX PATTERNS ==========
        # Backup for complex cases
        patterns = [
            r'cho\s+(tôi|em|mình)\s+(một|1)?\s*(bài|đề)',
            r'(tạo|ra|làm)\s+(cho\s+)?(tôi|em|mình)?\s*(một|1)?\s*(bài|đề)',
            r'(muốn|cần|được)\s+(làm|có)?\s*(bài|đề)',
        ]
        
        for pattern in patterns:
            if re.search(pattern, query_lower):
                print(f"   ✓ Matched regex pattern")
                return True
        
        print("   ✗ No quiz creation intent detected")
        return False
    
    def _extract_equation(self, query: str) -> Optional[str]:
        """Extract equation from query"""
        return extract_equation_from_query(query, self.client)
    
    def _extract_text_from_image(self, image_context: Dict) -> str:
        """
        Extract text from image using GPT-4 Vision
        
        Args:
            image_context: Dict with base64 image data
            
        Returns:
            Extracted text
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Trích xuất TOÀN BỘ nội dung văn bản trong ảnh.

    Yêu cầu:
    - Giữ nguyên format, xuống dòng
    - Bao gồm tất cả các lựa chọn A, B, C, D nếu có
    - Giữ nguyên ký hiệu đặc biệt (µm, %, →, v.v.)

    Chỉ trả về text được trích xuất, không thêm giải thích."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_context['base64']}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0
            )
            
            extracted = response.choices[0].message.content.strip()
            return extracted
            
        except Exception as e:
            print(f"   ⚠️ OCR error: {e}")
            return ""
    
    def _should_submit_quiz(self, user_query: str) -> bool:
        """
        Detect quiz submission intent
        
        Matches:
        - "nộp bài: 1-A,2-B,..."
        - "submit: 1-A,2-B,..."
        - "đáp án: 1-A,2-B,..."
        - "1-A,2-B,3-C,..." (bare answers)
        """
        query_lower = user_query.lower()
        
        # Check for submission keywords
        submission_keywords = [
            "nộp bài", "nộp đề", "nộp",
            "submit", "answer"
        ]
        
        for keyword in submission_keywords:
            if keyword in query_lower:
                print(f"   ✓ Submission keyword: '{keyword}'")
                return True
        
        # Check for answer pattern: "1-A,2-B,3-C,..."
        # Must have format: number-letter, at least 5 pairs
        answer_pattern = r'(\d+\s*-\s*[A-D]\s*,?\s*){5,}'
        if re.search(answer_pattern, user_query, re.IGNORECASE):
            print(f"   ✓ Answer pattern detected")
            return True
        
        return False
    
    def _should_view_quiz(self, user_query: str) -> bool:
        """
        Detect intent to view current quiz
        
        Matches:
        - "xem lại đề"
        - "nhắc lại đề"
        - "cho tôi xem đề"
        - "đề nào"
        - "show quiz"
        """
        query_lower = user_query.lower()
        
        # Keywords for viewing quiz
        view_keywords = [
            "xem lại đề", "nhắc lại đề", "xem đề", "hiển thị đề",
            "cho tôi xem đề", "cho em xem đề", "cho mình xem đề",
            "đề nào", "đề gì", "bài thi nào", "bài kiểm tra nào",
            "show quiz", "view quiz", "display quiz",
            "xem bài", "xem lại bài", "nhắc bài", "đọc lại đề"
        ]
        
        for keyword in view_keywords:
            if keyword in query_lower:
                print(f"   ✓ View quiz keyword: '{keyword}'")
                return True
        
        return False
    
    def _show_quiz_content(self, pending_quiz: Dict) -> str:
        """
        Return full quiz content with instructions
        
        Args:
            pending_quiz: Quiz data from database
            
        Returns:
            Formatted quiz markdown
        """
        quiz_id = pending_quiz.get("id")
        quiz_content = pending_quiz.get("content", "")
        subject = pending_quiz.get("subject", "N/A")
        topic = pending_quiz.get("topic", "N/A")
        
        if not quiz_content:
            return f"""⚠️ Không thể tải nội dung đề kiểm tra!

📋 **Thông tin đề:**
- Quiz ID: `{quiz_id}`
- Môn: {subject}
- Chủ đề: {topic}

💡 Vui lòng liên hệ giáo viên nếu vấn đề vẫn tiếp diễn."""
        
        return f"""📋 **ĐỀ KIỂM TRA ĐANG LÀM**

{quiz_content}

💡 **Để nộp bài, chat:**

Nộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B

⚠️ **Lưu ý:** Đảm bảo đúng 10 câu trước khi nộp!"""

    def _extract_answers(self, user_query: str) -> Optional[str]:
        """
        Extract answers from user query
        
        Input formats accepted:
        - "1-A,2-B,3-C,..."
        - "1-A, 2-B, 3-C, ..."
        - "1A,2B,3C,..."
        - "Nộp bài: 1-A,2-B,..."
        
        Returns:
            Normalized format "1-A,2-B,3-C,..." or None
        """
        try:
            # Remove submission keywords
            query = user_query
            for keyword in ["nộp bài:", "nộp:", "submit:"]:
                query = query.lower().replace(keyword, "")
            
            # Find all answer pairs
            # Pattern: number + optional dash/space + letter
            pattern = r'(\d+)\s*-?\s*([A-D])'
            matches = re.findall(pattern, query, re.IGNORECASE)
            
            if len(matches) < 10:
                print(f"   ⚠️ Only found {len(matches)} answers, need 10")
                return None
            
            # Normalize to "1-A,2-B,..." format
            normalized = []
            for num, letter in matches[:10]:  # Take first 10
                normalized.append(f"{num}-{letter.upper()}")
            
            result = ",".join(normalized)
            print(f"   ✓ Extracted answers: {result}")
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Error extracting answers: {e}")
            return None
    
    def query(
        self, 
        user_query: str, 
        conversation_history: List[Dict] = None,
        image_context: Optional[Dict] = None
    ) -> Dict:  # ← ĐỔI RETURN TYPE
        """
        Process user query with optional conversation history
        
        Args:
            user_query: Current user query
            conversation_history: Optional list of previous messages
            image_context: Optional dict with base64 image data
        
        Returns:
            {
                "response": str,
                "final_query": str
            }
        """
        
        messages = []
        final_query = user_query  # ← THÊM DÒNG NÀY
        
        try:
            print(f"\n{'='*70}")
            print(f"USER QUERY: {user_query}")
            print(f"{'='*70}")
            
            student_id = self.student_id if self.student_id else "unknown"

            if student_id == "unknown" and self.quiz_generator.student_profile:
                student_id = self.quiz_generator.student_profile.get("_id", "unknown")

            print(f"   🆔 Student ID: {student_id}")
            
            # ========== EXTRACT TEXT FROM IMAGE ==========
            if image_context:
                print("   🖼️  Detected image input")
                extracted_text = self._extract_text_from_image(image_context)
                
                if extracted_text:
                    print(f"   📝 Extracted {len(extracted_text)} chars from image")
                    user_query = extracted_text
                    final_query = extracted_text  # ← CẬP NHẬT final_query
                else:
                    print("   ⚠️  Could not extract text from image")
            
            # ========== CHECK SUBMISSION FIRST ==========
            if self._should_submit_quiz(user_query):
                print("   📝 Phát hiện ý định nộp bài!")
                
                pending_quiz = self.quiz_storage.get_latest_pending_quiz(student_id)
                
                if not pending_quiz:
                    return {
                        "response": """❌ Chưa có bài kiểm tra nào được tạo!

💡 Bạn có thể tạo đề mới bằng cách nói: "Tạo đề Toán về..."
""",
                        "final_query": final_query
                    }
                
                answers = self._extract_answers(user_query)
                
                if not answers:
                    return {
                        "response": f"""❌ Không thể đọc được đáp án!

📋 **Quiz đang làm:** `{pending_quiz['id']}`

💡 **Format đúng:**
- "Nộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B"
- "1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B"
- "1-A 2-B 3-C 4-D 5-A 6-B 7-C 8-D 9-A 10-B"

⚠️ **Lưu ý:** Cần đủ 10 câu, format: số-chữ cái (VD: 1-A, 2-B)""",
                        "final_query": final_query
                    }
                
                try:
                    quiz = self.quiz_storage.get_quiz(pending_quiz['id'])
                    
                    if not quiz:
                        return {
                            "response": f"❌ Lỗi: Không tìm thấy quiz {pending_quiz['id']}",
                            "final_query": final_query
                        }
                    
                    if self.submission_manager.check_quiz_submitted(pending_quiz['id'], student_id):
                        return {
                            "response": f"""❌ Bài này đã được nộp rồi!

📋 Quiz ID: `{pending_quiz['id']}`

💡 Bạn có thể tạo đề mới bằng cách nói: "Tạo đề Toán về..."
""",
                            "final_query": final_query
                        }
                    
                    answer_key = quiz.get("answer_key")
                    if not answer_key:
                        return {
                            "response": "❌ Lỗi: Đề thi thiếu đáp án. Vui lòng liên hệ admin.",
                            "final_query": final_query
                        }
                    
                    result = self.submission_manager.submit_quiz(
                        quiz_id=pending_quiz['id'],
                        student_id=student_id,
                        student_answers=answers,
                        answer_key=answer_key
                    )
                    
                    if not result["success"]:
                        return {
                            "response": f"❌ Lỗi nộp bài: {result.get('error', 'Unknown error')}",
                            "final_query": final_query
                        }
                    
                    self.quiz_storage.update_quiz_status(pending_quiz['id'], "completed")
                    
                    def call_daily(student_id: str, date: str):
                        import requests
                        api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8110')
                        try:
                            response = requests.get(
                                f"{api_base_url}/api/stats/daily",
                                params={"student_id": student_id, "date": date},
                                timeout=5
                            )
                            print(f"✅ Daily evaluation updated: {response.status_code}")
                        except Exception as e:
                            print(f"⚠️ Failed to call daily evaluation: {e}")

                    today = datetime.now().strftime("%Y-%m-%d")
                    threading.Thread(target=call_daily, args=(student_id, today), daemon=True).start()
                    
                    detailed = self.submission_manager.get_submission_with_details(
                        result["submission_id"],
                        answer_key
                    )

                    score = result["score"]
                    total = result["total"]
                    percentage = result["percentage"]

                    details_list = []
                    for detail in detailed["details"]:
                        num = detail["question_number"]
                        correct = detail["correct_answer"]
                        student = detail["student_answer"]
                        is_correct = detail["is_correct"]
                        
                        icon = "✅" if is_correct else "❌"
                        if is_correct:
                            line = f"   {icon} Câu {num}: {student} (Đúng)"
                        else:
                            line = f"   {icon} Câu {num}: {student} → Đúng là {correct}"
                        
                        details_list.append(line)

                    details_text = "\n".join(details_list)

                    return {
                        "response": f"""🎉 **ĐÃ NỘP BÀI THÀNH CÔNG!**

📊 **KẾT QUẢ:**
- Điểm: **{score}/{total}** ({percentage:.1f}%)
- Đúng: {detailed["correct_count"]} câu
- Sai: {detailed["incorrect_count"]} câu
- Thời gian hoàn thành: {result["duration"]} phút

📝 **CHI TIẾT:**
{details_text}

💾 **Thông tin:**
- Lần nộp thứ {result["daily_count"]} hôm nay

🎯 **Bạn có thể:**
- Tạo đề mới: "Tạo đề Toán về Hàm số"
""",
                        "final_query": final_query
                    }

                except Exception as e:
                    print(f"⚠️ Submission error: {e}")
                    return {
                        "response": f"❌ Lỗi khi nộp bài: {str(e)}",
                        "final_query": final_query
                    }

            # ========== CHECK PENDING QUIZ ==========
            pending_quiz = self.quiz_storage.get_latest_pending_quiz(student_id)

            if pending_quiz:
                print(f"\n⚠️  Student có quiz đang làm: {pending_quiz['id']}")
                print(f"   Input: {user_query}")
                
                if self._should_view_quiz(user_query):
                    print("   📋 Phát hiện ý định xem lại đề!")
                    return {
                        "response": self._show_quiz_content(pending_quiz),
                        "final_query": final_query
                    }
                
                if self._should_create_quiz(user_query):
                    print("   🚫 BLOCKED: Cannot create new quiz")
                    
                    return {
                        "response": f"""❌ Bạn không thể tạo đề mới khi đang có bài chưa nộp!

📋 **Bài kiểm tra chưa hoàn thành:**
- Môn: {pending_quiz.get('subject', 'N/A')}
- Chủ đề: {pending_quiz.get('topic', 'N/A')}

💡 **Bạn có thể:**
1. **Xem lại đề:** Gõ "Xem lại đề" hoặc "Nhắc lại đề"
2. **Nộp bài:** 
```
Nộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B
```

Sau khi nộp xong, bạn có thể tạo đề mới! 📝""",
                        "final_query": final_query
                    }
                
                guard_result = self.quiz_guard.is_cheating(user_query, pending_quiz)
                
                if guard_result["is_blocked"]:
                    print(f"   🚫 BLOCKED: {guard_result['reason']} (method: {guard_result['method']})")
                    
                    return {
                        "response": f"""🚫 **Không thể trả lời câu hỏi này!**

**Lý do:** {guard_result['reason']}

Bạn đang làm bài kiểm tra về **{pending_quiz.get('topic', 'N/A')}**.

💡 Hãy hoàn thành và nộp bài:
```
Nộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B
```
""",
                        "final_query": final_query
                    }
                else:
                    print(f"   ✓ ALLOWED: {guard_result['reason']} (method: {guard_result['method']})")
            
            # ========== CHECK IF QUIZ REQUEST ==========
            if self._should_create_quiz(user_query):
                print("\n📝 Phát hiện yêu cầu tạo đề kiểm tra!")
                
                quiz_info = extract_topic_from_query(user_query, self.client)
                
                if not quiz_info:
                    return {
                        "response": """⚠️ Không thể hiểu yêu cầu của bạn.

💡 Vui lòng thử lại với format rõ ràng hơn:
- "Tạo đề [Môn] về [Chủ đề]"
- "Ra đề kiểm tra [Môn] về [Chủ đề]"

📚 **Các môn hỗ trợ:** Toán, Vật lý, Hóa học, Sinh học""",
                        "final_query": final_query
                    }

                if not quiz_info.get("subject"):
                    return {
                        "response": """⚠️ Không xác định được môn học.

💡 **Các môn hỗ trợ:** Toán, Vật lý, Hóa học, Sinh học

**Ví dụ câu hỏi đúng:**
- "Tạo đề Toán về Hàm số bậc hai"
- "Đề kiểm tra Vật lý về Dao động điều hòa"
- "Ra 10 câu Hóa về Axit - Bazơ - Muối"
""",
                        "final_query": final_query
                    }

                detected_subject = quiz_info.get("subject")
                if detected_subject not in ALLOWED_QUIZ_SUBJECTS:
                    return {
                        "response": f"""⚠️ Xin lỗi, hiện tại hệ thống chỉ hỗ trợ **4 môn tự nhiên**.

🔍 **Bạn yêu cầu:** {detected_subject}

📚 **Các môn được hỗ trợ:**
✅ Toán
✅ Vật lý
✅ Hóa học
✅ Sinh học

❌ **Không hỗ trợ:** Văn, Sử, Địa, Anh, Tin, v.v.

💡 **Bạn có thể thử:**
- "Tạo đề Toán về Hệ bất phương trình"
- "Tạo đề Vật lý về Động lực học"
- "Tạo đề Hóa học về Bảng tuần hoàn"
- "Tạo đề Sinh học về Quang hợp"
""",
                        "final_query": final_query
                    }
                
                if not quiz_info.get("topic") or len(quiz_info.get("topic", "").strip()) < 3:
                    return {
                        "response": f"""⚠️ Vui lòng chỉ rõ chủ đề cần tạo đề.

📚 **Môn:** {detected_subject}

💡 **Ví dụ:**
- "Tạo đề {detected_subject} về [Chủ đề cụ thể]"

**Gợi ý chủ đề:**
- Tạo đề Toán về Hàm số bậc hai
- Tạo đề Vật lý về Dao động điều hòa
- Tạo đề Hóa học về Axit-Bazơ""",
                        "final_query": final_query
                    }

                print(f"   📚 Môn: {quiz_info['subject']}")
                print(f"   📖 Chủ đề: {quiz_info['topic']}")
                
                user_difficulty = quiz_info.get("user_difficulty")
                
                if user_difficulty:
                    print(f"   🎯 Độ khó user chỉ định: {user_difficulty}")
                    use_student_difficulty = False
                else:
                    print(f"   🎯 Sử dụng độ khó từ profile")
                    use_student_difficulty = True
                
                result = self.quiz_generator.generate_quiz(
                    subject=quiz_info["subject"],
                    topic=quiz_info["topic"],
                    difficulty=user_difficulty,
                    use_student_difficulty=use_student_difficulty
                )
                
                if result["success"]:
                    try:
                        if not result.get("answer_key"):
                            print("   ⚠️ Thiếu answer_key!")
                            return {
                                "response": "❌ Lỗi: Không thể tạo đề vì thiếu đáp án. Vui lòng thử lại.",
                                "final_query": final_query
                            }
                        
                        quiz_id = self.quiz_storage.save_quiz(
                            student_id=student_id,
                            content=result['quiz_markdown'],
                            answer_key=result['answer_key'],
                            subject=quiz_info["subject"],
                            topic=quiz_info["topic"],
                            difficulty=result["metadata"]["difficulty"]
                        )
                        
                        print(f"✅ Đã lưu vào database với ID: {quiz_id}")
                    except Exception as e:
                        print(f"⚠️ Không thể lưu quiz: {e}")
                    
                    return {
                        "response": f"""✅ Đã tạo xong đề kiểm tra!

{result['quiz_markdown']}

💡 **Để nộp bài:**
```
Nộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B
```
""",
                        "final_query": final_query
                    }
                else:
                    return {
                        "response": f"""❌ Không thể tạo đề kiểm tra: {result['error']}

💡 Vui lòng thử lại hoặc cung cấp thông tin rõ ràng hơn.""",
                        "final_query": final_query
                    }
            
            # ========== CHECK IF GRAPH REQUEST ==========
            if self._should_draw_graph(user_query):
                print("\n📊 Phát hiện yêu cầu vẽ đồ thị!")
                
                equation = self._extract_equation(user_query)
                
                if not equation:
                    return {
                        "response": "⚠️ Không thể xác định hàm số cần vẽ. Vui lòng nhập rõ hơn (VD: 'vẽ đồ thị y = x**2')",
                        "final_query": final_query
                    }
                
                print(f"   📝 Equation: y = {equation}")
                
                x_min, x_max = extract_range_from_query(user_query)
                print(f"   📏 Range: [{x_min}, {x_max}]")
                
                result = self.graph_generator.generate_graph(equation, x_min, x_max)
                
                if result["success"]:
                    return {
                        "response": f"""✅ Đã vẽ xong đồ thị!

📊 Thông tin:
- Hàm số: y = {equation}
- Khoảng giá trị: x ∈ [{x_min}, {x_max}]
- File: {result['file_path']}
- Kích thước: {result['file_size']/1024:.1f}KB

[IMAGE:{result['file_path']}]

💡 Bạn có muốn tôi giải thích gì về đồ thị này không?""",
                        "final_query": final_query
                    }
                else:
                    return {
                        "response": f"""❌ Không thể vẽ đồ thị: {result['error']}

💡 Gợi ý:
- Kiểm tra cú pháp hàm số (VD: x**2, sin(x), 2*x + 3)
- Đảm bảo hàm số hợp lệ trong khoảng [{x_min}, {x_max}]
- Thử lại với hàm số đơn giản hơn""",
                        "final_query": final_query
                    }
            
            # ========== DECIDE IF SHOULD USE SEARCH TOOL ==========
            should_search = self._should_use_tool(user_query)
    
            if should_search:
                print("\n🔧 Quyết định: Sử dụng tool search_questions")
                
                results = self.retriever.search(
                    query=user_query,
                    subject=None,
                    top_k=3
                )
                
                # ========== FALLBACK IF SCORE TOO LOW ==========
                if not results or results[0]['score'] < 0.8:
                    print(f"   ✗ Score too low → FALLBACK TO CHAT")
                    
                    messages = [
                        {
                            "role": "system",
                            "content": self._get_system_prompt(mode="general")
                        }
                    ]
                    
                    if conversation_history:
                        recent_history = conversation_history[-10:]
                        messages.extend(recent_history)
                        print(f"   📜 Added {len(recent_history)} history messages")
                        
                    messages.append({
                        "role": "user",
                        "content": user_query
                    })
                    
                    response = self.client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    return {
                        "response": response.choices[0].message.content,
                        "final_query": final_query
                    }
                
                # ========== EXTRACT BEST RESULT ==========
                best_result = results[0]
                best_id = best_result['question_id']
                best_answer = best_result['correct_answer']
                best_answer_text = best_result['correct_answer_text']
                best_explanation = best_result.get('explanation', '')
                
                print(f"   ✓ Best match: {best_id} (score: {best_result['score']:.2f})")
                print(f"   ✓ Answer: {best_answer}")
                print(f"   ✓ Explanation length: {len(best_explanation)} chars")
                
                # ========== IF HAS EXPLANATION → RETURN DIRECTLY ==========
                if best_explanation:
                    formatted_response = f"""**Đáp án {best_answer}: {best_answer_text}**

**Giải thích:**
{best_explanation}"""
                    
                    return {
                        "response": formatted_response,
                        "final_query": final_query
                    }
                
                # ========== NO EXPLANATION → USE LLM ==========
                messages = [
                    {
                        "role": "system",
                        "content": self._get_system_prompt(mode="search")
                    }
                ]
                
                if conversation_history:
                    recent_history = conversation_history[-10:]
                    messages.extend(recent_history)
                    print(f"   📜 Added {len(recent_history)} history messages")
                
                user_content = f"""Học sinh hỏi: {user_query}

ĐÁP ÁN ĐÚNG: {best_answer}: {best_answer_text}

YÊU CẦU:
- Giải thích TẠI SAO đáp án này đúng (3-5 câu)
- Tập trung vào logic của câu hỏi
- Ngắn gọn, dễ hiểu

ĐỊNH DẠNG:
**Đáp án {best_answer}: {best_answer_text}**

**Giải thích:**
[3-5 câu giải thích]"""

                messages.append({
                    "role": "user",
                    "content": user_content
                })

            else:
                print("\n💬 Quyết định: Trả lời trực tiếp (không cần search)")
                
                messages = [
                    {
                        "role": "system",
                        "content": self._get_system_prompt(mode="general")
                    }
                ]
                
                if conversation_history:
                    recent_history = conversation_history[-10:]
                    messages.extend(recent_history)
                    print(f"   📜 Added {len(recent_history)} history messages")
                
                messages.append({
                    "role": "user",
                    "content": user_query
                })

            # ========== GET LLM RESPONSE ==========
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=2000
            )

            return {
                "response": response.choices[0].message.content,
                "final_query": final_query
            }
            
        except Exception as e:
            return {
                "response": f"⚠️ Lỗi xử lý câu hỏi: {str(e)}",
                "final_query": final_query
            }

# ================== RAG SYSTEM ==================
class ScienceQASystem:
    def __init__(self, student_id: str = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.intent_classifier = IntentClassifier(self.client)
        self.retriever = QuestionRetriever(self.client, QDRANT_PATH, COLLECTION_NAME)
        self.agent = SimpleAgent(self.client, self.intent_classifier, self.retriever, student_id)
    
    def query(
        self, 
        user_query: str, 
        conversation_history: List[Dict] = None,
        image_context: Optional[Dict] = None
    ) -> Dict:
        """
        Process user query through RAG system with optional conversation history
        
        Args:
            user_query: Current user query
            conversation_history: Optional list of previous messages
            image_context: Optional dict with base64 image data
            
        Returns:
            Response string
        """
        return self.agent.query(user_query, conversation_history, image_context)

# ================== DISPLAY HELPER ==================
def display_response(response: str):
    """Display response with image support"""
    
    # Check for image tag
    image_pattern = r'\[IMAGE:(.+?)\]'
    match = re.search(image_pattern, response)
    
    if match:
        img_path = match.group(1)
        
        # Remove image tag from text
        text = response.replace(match.group(0), '')
        print(text)
        
        # Try to open image
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', img_path], check=False)
            elif platform.system() == 'Linux':
                subprocess.run(['xdg-open', img_path], check=False)
            elif platform.system() == 'Windows':
                os.startfile(img_path)
            
            print(f"\n🖼️  Đã mở ảnh: {img_path}")
        except Exception as e:
            print(f"\n⚠️  Không thể mở ảnh tự động: {e}")
            print(f"   Vui lòng mở file: {img_path}")
    else:
        print(response)


# ================== MAIN CLI ==================
def main():
    print("=" * 70)
    print("HỆ THỐNG RAG - TRỢ LÝ HỌC TẬP MÔN TỰ NHIÊN")
    print("=" * 70)
    print("Môn học hỗ trợ: Toán, Lý, Hóa, Sinh")
    print("✨ Tính năng: Vẽ đồ thị + Tạo đề kiểm tra + Chấm điểm tự động")
    print("Gõ 'exit' hoặc 'quit' để thoát")
    print("=" * 70)
    
    # Initialize system
    print("\n🔧 Đang khởi tạo hệ thống...")
    try:
        rag_system = ScienceQASystem()
        print("✅ Hệ thống sẵn sàng!\n")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return
    
    # Show examples
    print("💡 Ví dụ câu hỏi:")
    print("   - Định luật Newton là gì?")
    print("   - Vẽ đồ thị y = x**2")
    print("   - Vẽ đồ thị sin(x) từ -5 đến 5")
    print("   - Tạo đề kiểm tra Vật lý về Động lực học")
    print("   - Tạo đề Toán về Hệ bất phương trình")
    print("   - Hàm bậc hai có tính chất gì?\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n🎓 Học sinh: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'thoát']:
                print("\n👋 Tạm biệt! Chúc bạn học tốt!")
                break
            
            # Process query
            response = rag_system.query(user_input)
            
            print(f"\n🤖 Trợ lý:")
            display_response(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n⚠️ Lỗi: {e}")

if __name__ == "__main__":
    main()