"""
src/tools/quiz_generator.py

Tạo đề kiểm tra trắc nghiệm từ nội dung tài liệu (RAG-based).
Câu hỏi được sinh dựa trên chunks thực tế từ tài liệu điều chế số,
không sinh từ kiến thức chung của LLM.
"""

import re
import os
import sqlite3
from typing import Dict, Optional
from openai import OpenAI

from src.tools.rag_tool import get_rag_tool


# ================== STUDENT PROFILE ==================

def load_student_profile(student_id: str) -> Dict:
    """Load student difficulty level from evaluation database."""
    db_path = "database/student_evaluations.db"

    default = {"_id": student_id, "difficulty_level": "medium"}

    if not os.path.exists(db_path):
        print("⚠️  Evaluation DB not found, using medium difficulty")
        return default

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM daily_evaluations WHERE student_id = ? ORDER BY date DESC LIMIT 1",
            (student_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            print("⚠️  No evaluation found, using medium difficulty")
            return default

        rating = dict(row).get("rating", "")
        if "Xuất sắc" in rating or "Giỏi" in rating:
            level = "hard"
        elif "Yếu" in rating:
            level = "easy"
        else:
            level = "medium"

        print(f"   ✓ Profile loaded: {rating} → difficulty: {level}")
        return {"_id": student_id, "difficulty_level": level, "rating": rating}

    except Exception as e:
        print(f"⚠️  Profile load error: {e}")
        return default


def _to_vietnamese_difficulty(difficulty: str) -> str:
    return {"easy": "dễ", "medium": "trung bình", "hard": "khó"}.get(
        difficulty.lower(), "trung bình"
    )


# ================== QUIZ GENERATOR ==================

class QuizGenerator:
    """Generate quiz questions from document chunks (RAG-based).

    Flow:
        topic
          → RAGTool.retrieve_by_topic()   (k=10 most relevant chunks)
          → build context string
          → LLM generates 10 questions BASED ON the context
          → parse & validate answer key
    """

    def __init__(self, client: OpenAI, student_id: str = None, api_base_url: str = None):
        self.client = client
        self.student_id = student_id
        self.student_profile = None  # lazy loaded
        self.rag = get_rag_tool()

    # ────────────────────────────────────────────────────────────────
    # Public
    # ────────────────────────────────────────────────────────────────

    def generate_quiz(
        self,
        subject: str,
        topic: str,
        difficulty: str = None,
        use_student_difficulty: bool = True,
    ) -> Dict:
        """Generate a 10-question multiple-choice quiz from document content.

        Returns:
            {
                "success": bool,
                "quiz_markdown": str,
                "answer_key": str,       # "1-A,2-B,..."
                "metadata": dict,
            }
        """
        self._ensure_profile_loaded()

        # Resolve difficulty
        if use_student_difficulty or difficulty is None:
            level_vi = _to_vietnamese_difficulty(
                (self.student_profile or {}).get("difficulty_level", "medium")
            )
        else:
            level_vi = _to_vietnamese_difficulty(difficulty)

        print(f"\n📝 Tạo đề: {topic} | Độ khó: {level_vi}")

        # ── 1. Retrieve relevant chunks ──────────────────────────────
        chunks = self.rag.retrieve_by_topic(topic, k=10)
        if not chunks:
            return {"success": False, "error": "Không tìm thấy nội dung liên quan trong tài liệu."}

        context = self._build_quiz_context(chunks)
        print(f"   ✓ Retrieved {len(chunks)} chunks for context")

        # ── 2. Generate quiz from context ────────────────────────────
        quiz_markdown, answer_key = self._generate_from_context(topic, level_vi, context)

        if not answer_key:
            print("   ⚠️  Answer key missing, retrying...")
            quiz_markdown, answer_key = self._generate_from_context(
                topic, level_vi, context, force_answer_key=True
            )

        if not self._validate_quiz(quiz_markdown):
            print("   ⚠️  Quiz format invalid, retrying...")
            quiz_markdown, answer_key = self._generate_from_context(
                topic, level_vi, context, strict_format=True
            )

        if not answer_key:
            return {"success": False, "error": "Không thể tạo đáp án hợp lệ. Vui lòng thử lại."}

        print("   ✓ Quiz generated successfully")

        return {
            "success": True,
            "quiz_markdown": quiz_markdown,
            "answer_key": answer_key,
            "metadata": {
                "subject": subject,
                "topic": topic,
                "difficulty": level_vi,
                "num_questions": 10,
                "time_limit": 15,
                "chunks_used": [c["chunk_id"] for c in chunks],
            },
        }

    # ────────────────────────────────────────────────────────────────
    # Private
    # ────────────────────────────────────────────────────────────────

    def _ensure_profile_loaded(self):
        if self.student_id and not self.student_profile:
            self.student_profile = load_student_profile(self.student_id)

    def _build_quiz_context(self, chunks: list) -> str:
        """Format chunks into a context block for the LLM."""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Section {c['section']}: {c['section_title']}\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate_from_context(
        self,
        topic: str,
        difficulty_vi: str,
        context: str,
        force_answer_key: bool = False,
        strict_format: bool = False,
    ):
        """Call LLM to generate quiz. Returns (quiz_markdown, answer_key)."""

        extra = ""
        if force_answer_key:
            extra += "\n\n⚠️ BẮT BUỘC: Thêm dòng ẩn ở cuối:\n<!-- ANSWER_KEY: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B -->"
        if strict_format:
            extra += "\n\n⚠️ Đảm bảo đúng format ## **Câu X**: và **A.**, **B.**, **C.**, **D.**"

        system_prompt = f"""Bạn là chuyên gia ra đề kiểm tra về điều chế số (digital modulation & communications).

Nhiệm vụ: Tạo 10 câu trắc nghiệm dựa HOÀN TOÀN vào nội dung tài liệu được cung cấp.

QUY TẮC QUAN TRỌNG:
1. Chỉ ra câu hỏi từ thông tin CÓ TRONG TÀI LIỆU, không thêm kiến thức ngoài
2. Đúng 10 câu, mỗi câu 4 đáp án (A/B/C/D)
3. Không để đáp án đúng trong nội dung đề bài
4. Đáp án nhiễu phải hợp lý, không quá dễ đoán

ĐỘ KHÓ "{difficulty_vi}":
- dễ: Câu hỏi định nghĩa, nhận biết khái niệm trực tiếp từ tài liệu
- trung bình: So sánh, giải thích, áp dụng khái niệm
- khó: Phân tích, kết hợp nhiều khái niệm, tính toán có bẫy

FORMAT BẮT BUỘC:
# ĐỀ KIỂM TRA 15 PHÚT - ĐIỀU CHẾ SỐ
**Chủ đề**: {topic}
**Độ khó**: {difficulty_vi}
**Thời gian**: 15 phút

---

## **Câu 1**: [câu hỏi]
**A.** [đáp án]
**B.** [đáp án]
**C.** [đáp án]
**D.** [đáp án]

[... Câu 2-10 tương tự ...]

---
_Hết_

<!-- ANSWER_KEY: 1-X,2-Y,3-Z,4-W,5-V,6-U,7-T,8-S,9-R,10-Q -->

Trong đó X,Y,...Q là đáp án đúng (A/B/C/D) của mỗi câu."""

        user_prompt = f"""Tài liệu tham khảo:

{context}

---

Yêu cầu: Tạo 10 câu trắc nghiệm về chủ đề "{topic}", độ khó {difficulty_vi}, dựa vào nội dung tài liệu trên.{extra}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=3000,
            )
            quiz_markdown = response.choices[0].message.content.strip()
            answer_key = self._extract_answer_key(quiz_markdown)
            return quiz_markdown, answer_key

        except Exception as e:
            print(f"   ✗ Generation error: {e}")
            return "", None

    def _validate_quiz(self, markdown: str) -> bool:
        """Check that quiz has 10 questions with correct format."""
        questions = re.findall(r"##\s+\*\*Câu\s+\d+\*\*:", markdown)
        if len(questions) != 10:
            print(f"   ⚠️ Found {len(questions)}/10 questions")
            return False
        options = re.findall(r"\*\*[A-D]\.\*\*", markdown)
        if len(options) != 40:
            print(f"   ⚠️ Found {len(options)}/40 options")
            return False
        return True

    def _extract_answer_key(self, markdown: str) -> Optional[str]:
        """Extract answer key from quiz markdown. Returns '1-A,2-B,...' or None."""
        # Pattern 1: <!-- ANSWER_KEY: 1-A,2-B,... -->
        m = re.search(
            r"<!--\s*ANSWER_KEY:\s*([0-9]+-[A-D](?:,\s*[0-9]+-[A-D])+)\s*-->",
            markdown,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).replace(" ", "")

        # Pattern 2: "Đáp án: 1-A,2-B,..."
        m = re.search(
            r"(?:Đáp án:|Answer key:)\s*([0-9]+-[A-D](?:,\s*[0-9]+-[A-D])+)",
            markdown,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).replace(" ", "")

        # Pattern 3: list "1. A\n2. B\n..."
        m = re.search(
            r"\*\*Đáp án:\*\*\s*\n((?:\d+\.\s*[A-D]\s*\n?)+)",
            markdown,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            answers = []
            for line in m.group(1).strip().split("\n"):
                lm = re.match(r"(\d+)\.\s*([A-D])", line.strip())
                if lm:
                    answers.append(f"{lm.group(1)}-{lm.group(2)}")
            if len(answers) == 10:
                return ",".join(answers)

        # Pattern 4: standalone "1. A" lines
        answers, lines = [], markdown.split("\n")
        for line in lines:
            lm = re.match(r"^\s*(\d+)\.\s*([A-D])\s*$", line.strip())
            if lm:
                answers.append(f"{lm.group(1)}-{lm.group(2)}")
                if len(answers) == 10:
                    return ",".join(answers)
            elif answers:
                answers = []  # reset on non-matching line

        return None

    def _extract_metadata(self, markdown: str) -> Dict:
        return {"total_questions_found": len(re.findall(r"##\s+\*\*Câu\s+\d+\*\*:", markdown))}
