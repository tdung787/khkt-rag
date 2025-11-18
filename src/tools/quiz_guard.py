"""
tools/quiz_guard.py

Bảo vệ quiz khỏi gian lận - chặn câu hỏi liên quan đến đề đang làm
"""

import re
from typing import Dict, Optional, List
from openai import OpenAI


class QuizGuard:
    """Guard system to prevent cheating during quiz"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.cache = {}  # Cache LLM results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)

    def _extract_all_questions(self, content: str) -> List[str]:
        """Extract all question texts from quiz"""
        try:
            pattern = r'##\s+\*\*Câu\s+\d+\*\*:\s*(.+?)(?=\*\*[A-D]\.\*\*|$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            questions = []
            for match in matches:
                question_text = ' '.join(match.strip().split())
                questions.append(question_text)
            
            return questions
        except Exception as e:
            print(f"⚠️ Extract error: {e}")
            return []

    def is_cheating(self, user_query: str, current_quiz: Dict) -> Dict:
        """3-layer detection: explicit → similarity → LLM"""
        
        # Layer 1: Explicit
        if self._has_explicit_cheating(user_query):
            return {
                "is_blocked": True,
                "reason": "Câu hỏi trực tiếp về đề thi",
                "confidence": 1.0,
                "method": "explicit"
            }
        
        # Layer 2: Similarity
        quiz_questions = self._extract_all_questions(current_quiz['content'])
        
        max_similarity = 0.0
        for q_text in quiz_questions:
            similarity = self._calculate_text_similarity(user_query, q_text)
            max_similarity = max(max_similarity, similarity)
            
            if similarity > 0.6:  # 60% threshold
                return {
                    "is_blocked": True,
                    "reason": f"Câu hỏi trùng {int(similarity*100)}% với câu trong đề",
                    "confidence": 0.98,
                    "method": "similarity"
                }
        
        print(f"   📊 Max similarity: {max_similarity:.2f}")
        
        # Layer 3: LLM
        return self._llm_classify(user_query, current_quiz)
    
    def _has_explicit_cheating(self, query: str) -> bool:
        """
        Check for explicit cheating patterns
        
        Patterns:
        - "câu 3", "câu số 5"
        - "đáp án"
        - "bài này", "bài kiểm tra"
        - "chọn A/B/C/D"
        """
        query_lower = query.lower()
        
        # Explicit patterns
        patterns = [
            r'câu\s+\d+',              # "câu 3"
            r'câu\s+số\s+\d+',         # "câu số 5"
            r'đáp\s*án',               # "đáp án"
            r'chọn\s+[A-D]',           # "chọn A"
            r'bài\s+(này|đó|kiểm\s*tra)',  # "bài này", "bài kiểm tra"
            r'đề\s+(này|đó|thi)',      # "đề này"
            r'câu\s+hỏi\s+số',         # "câu hỏi số"
        ]
        
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _llm_classify(self, query: str, quiz: Dict) -> Dict:
        """
        Use LLM to classify - IMPROVED with stricter prompt
        
        Cost: ~$0.0002/call
        """
        cache_key = f"{quiz['id']}:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Extract first 5 questions for better context
            first_questions = self._extract_first_questions(quiz['content'], count=5)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là hệ thống phát hiện gian lận thi cử CỰC KỲ NGHIÊM NGẶT.

    NHIỆM VỤ: Xác định câu hỏi của học sinh có giúp họ làm bài kiểm tra không.

    ⚠️ QUAN TRỌNG NHẤT:
    Nếu câu hỏi của học sinh TRÙNG KHỚP >50% với bất kỳ câu nào trong đề thi
    → PHẢI BLOCK (trả về YES)

    LIÊN QUAN (BLOCK) bao gồm:
    1. ✅ Copy y nguyên câu hỏi trong đề (dù chỉ thay đổi vài từ)
    2. ✅ Hỏi về kiến thức TRỰC TIẾP có trong đề
    3. ✅ Hỏi cách giải dạng bài CHÍNH XÁC trong đề
    4. ✅ Hỏi "ví dụ" mà ví dụ đó chính là câu trong đề
    5. ✅ Hỏi với ngữ cảnh giống hệt câu trong đề

    KHÔNG LIÊN QUAN (ALLOW):
    1. ❌ Hỏi định nghĩa/khái niệm TỔNG QUÁT (không cụ thể như đề)
    2. ❌ Hỏi về chủ đề KHÁC với đề
    3. ❌ Hỏi câu chuyện cá nhân, thời tiết

    PHƯƠNG PHÁP ĐÁNH GIÁ:
    1. So sánh VĂN BẢN: Câu hỏi có trùng >50% với câu nào trong đề?
    2. So sánh NGỮ NGHĨA: Nội dung có giống câu nào trong đề?
    3. So sánh BỐI CẢNH: Ví dụ/số liệu có trùng với đề?

    🚨 NẾU NGHI NGỜ → BLOCK (thà nhầm còn hơn để gian lận)

    Trả lời: CHỈ "YES" (block) hoặc "NO" (allow)"""
                    },
                    {
                        "role": "user",
                        "content": f"""Đề kiểm tra đang làm:
    📚 Môn: {quiz.get('subject', 'N/A')}
    📖 Chủ đề: {quiz.get('topic', 'N/A')}

    🔍 Nội dung một số câu trong đề:
    {first_questions}

    ❓ Câu hỏi của học sinh:
    "{query}"

    ⚠️ PHÂN TÍCH:
    1. So sánh văn bản: Câu hỏi có trùng khớp với câu nào trong đề không?
    2. Độ tương đồng: Ước tính % trùng lặp nội dung
    3. Quyết định: Nếu >50% trùng → YES, ngược lại → NO

    → Trả lời: YES hoặc NO"""
                    }
                ],
                temperature=0,  # ← Đặt 0 để deterministic
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            is_blocked = "YES" in answer
            
            result = {
                "is_blocked": is_blocked,
                "reason": "Câu hỏi trùng hoặc giống với nội dung đề thi" if is_blocked else "Câu hỏi không liên quan đến đề",
                "confidence": 0.95,
                "method": "llm"
            }
            
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"⚠️ LLM classify error: {e}")
            # Default: block if error (an toàn hơn)
            return {
                "is_blocked": True,
                "reason": "Không thể xác định (lỗi hệ thống) - Block để an toàn",
                "confidence": 0.5,
                "method": "error"
            }
    
    def _extract_first_questions(self, content: str, count: int = 3) -> str:
        """Extract first N questions from quiz content"""
        try:
            # Find questions using pattern: ## **Câu X**:
            pattern = r'##\s+\*\*Câu\s+\d+\*\*:.+?(?=##\s+\*\*Câu\s+\d+\*\*:|---|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            # Return first N questions
            questions = matches[:count]
            return '\n\n'.join(questions) if questions else "Không trích xuất được nội dung"
            
        except Exception as e:
            return f"Lỗi trích xuất: {e}"