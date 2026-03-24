import os
import re
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

from src.tools.rag_tool import get_rag_tool
from src.tools.graph_generator import GraphGenerator
from src.tools.quiz_generator import QuizGenerator
from src.tools.quiz_storage import QuizStorage
from src.tools.quiz_guard import QuizGuard
from src.tools.submission_manager import SubmissionManager

load_dotenv()

# ================== CONFIG ==================
OPENAI_MODEL = "gpt-4o"


# ================== ROLE CHECK ==================
def get_user_role(user_id: str) -> Optional[str]:
    """Return 'student', 'teacher', or None."""
    try:
        api_base = os.getenv("EXTERNAL_API_BASE_URL", "http://localhost:8222")

        try:
            r = requests.get(f"{api_base}/api/public/rag/students", timeout=5)
            if r.status_code == 200:
                for s in r.json().get("data", {}).get("students", []):
                    if s.get("user_id", {}).get("_id") == user_id:
                        return "student"
        except Exception:
            pass

        try:
            r = requests.get(f"{api_base}/api/public/rag/teachers", timeout=5)
            if r.status_code == 200:
                for t in r.json().get("data", []):
                    if t.get("_id") == user_id:
                        return "teacher"
        except Exception:
            pass

    except Exception:
        pass

    return None


# ================== TOOL DEFINITIONS ==================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_answer",
            "description": (
                "Trả lời câu hỏi về điều chế số (digital modulation) từ tài liệu kỹ thuật. "
                "Dùng khi người dùng hỏi về khái niệm, kỹ thuật, công thức, so sánh, "
                "hoặc bất kỳ nội dung kỹ thuật nào liên quan đến điều chế số."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu hỏi kỹ thuật cần tra cứu trong tài liệu",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_quiz",
            "description": (
                "Tạo bài kiểm tra trắc nghiệm về điều chế số. "
                "Dùng khi người dùng yêu cầu tạo đề, ra đề, kiểm tra, bài thi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Chủ đề cụ thể cần tạo đề (ví dụ: FSK, QPSK, BER, Nyquist filter)",
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                        "description": "Độ khó của đề (mặc định medium nếu không rõ)",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_quiz",
            "description": (
                "Nộp bài kiểm tra và chấm điểm. "
                "Dùng khi người dùng gửi đáp án theo format '1-A,2-B,...' hoặc nói 'nộp bài'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answers_text": {
                        "type": "string",
                        "description": "Chuỗi đáp án đầy đủ từ người dùng",
                    }
                },
                "required": ["answers_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_quiz",
            "description": (
                "Xem lại đề bài kiểm tra đang làm. "
                "Dùng khi người dùng muốn xem lại đề, nhắc lại đề."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_graph",
            "description": (
                "Vẽ đồ thị hàm số toán học. "
                "Dùng khi người dùng yêu cầu vẽ đồ thị, plot hàm số."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "Hàm số cần vẽ theo Python syntax (ví dụ: x**2, np.sin(x), 2*x+3)",
                    },
                    "x_min": {
                        "type": "number",
                        "description": "Giá trị x nhỏ nhất (mặc định -10)",
                    },
                    "x_max": {
                        "type": "number",
                        "description": "Giá trị x lớn nhất (mặc định 10)",
                    },
                },
                "required": ["equation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_chat",
            "description": (
                "Trả lời tin nhắn thông thường, chào hỏi, cảm ơn, hoặc hội thoại không liên quan kỹ thuật."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Tin nhắn của người dùng",
                    }
                },
                "required": ["message"],
            },
        },
    },
]


# ================== AGENT ==================
class DigitalModulationAgent:
    """LLM-based agent for digital modulation chatbot.

    Uses OpenAI function calling to decide which tool to invoke.
    No hardcoded keyword routing.
    """

    def __init__(self, student_id: str = None):
        self.client = OpenAI()
        self.student_id = student_id or "unknown"

        # Tools
        self.rag = get_rag_tool()
        self.graph_generator = GraphGenerator(self.client)
        self.quiz_generator = QuizGenerator(self.client, student_id=student_id)
        self.quiz_storage = QuizStorage()
        self.quiz_guard = QuizGuard(self.client)
        self.submission_manager = SubmissionManager()

    # ────────────────────────────────────────────────────────────────
    # Public entry point
    # ────────────────────────────────────────────────────────────────
    def query(
        self,
        user_query: str,
        conversation_history: List[Dict] = None,
        image_context: Optional[Dict] = None,
    ) -> Dict:
        """Process a user message and return {"response": str, "final_query": str}."""

        final_query = user_query
        student_id = self.student_id

        print(f"\n{'='*70}")
        print(f"USER QUERY: {user_query}")
        print(f"STUDENT ID: {student_id}")
        print(f"{'='*70}")

        try:
            # ── 1. OCR image if present ──────────────────────────────
            if image_context:
                extracted = self._extract_text_from_image(image_context)
                if extracted:
                    user_query = f"{user_query}\n\n{extracted}"
                    final_query = extracted

            # ── 2. Build messages for LLM ────────────────────────────
            system_prompt = self._build_system_prompt(student_id)
            messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                recent = conversation_history[-10:]
                for msg in recent:
                    if msg["role"] == "user":
                        messages.append({
                            "role": "user",
                            "content": self._extract_hidden_text(msg["content"]),
                        })
                    else:
                        messages.append(msg)

            messages.append({"role": "user", "content": user_query})

            # ── 3. LLM decides which tool to call ────────────────────
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )

            msg = response.choices[0].message

            # No tool call → direct LLM response
            if not msg.tool_calls:
                print("   → LLM chose: direct response")
                return {"response": msg.content or "", "final_query": final_query}

            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"   → LLM chose tool: {tool_name}")
            print(f"   → Args: {tool_args}")

            # ── 4. Execute chosen tool ───────────────────────────────
            result = self._execute_tool(
                tool_name, tool_args, user_query, student_id
            )
            return {"response": result, "final_query": final_query}

        except Exception as e:
            print(f"⚠️ Agent error: {e}")
            return {
                "response": f"⚠️ Lỗi xử lý: {str(e)}",
                "final_query": final_query,
            }

    # ────────────────────────────────────────────────────────────────
    # Tool executor
    # ────────────────────────────────────────────────────────────────
    def _execute_tool(
        self, tool_name: str, args: Dict, user_query: str, student_id: str
    ) -> str:

        # ── rag_answer ───────────────────────────────────────────────
        if tool_name == "rag_answer":
            # Check quiz guard first
            pending = self.quiz_storage.get_latest_pending_quiz(student_id)
            if pending:
                user_role = get_user_role(student_id)
                guard = self.quiz_guard.is_cheating(user_query, pending, user_role)
                if guard["is_blocked"]:
                    print(f"   🚫 Quiz guard blocked: {guard['reason']}")
                    return (
                        f"🚫 **Không thể trả lời câu hỏi này!**\n\n"
                        f"**Lý do:** {guard['reason']}\n\n"
                        f"Bạn đang làm bài kiểm tra về **{pending.get('topic', 'N/A')}**.\n\n"
                        "💡 Hãy hoàn thành và nộp bài:\n"
                        "```\nNộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B\n```"
                    )

            return self.rag.answer(args["query"])

        # ── create_quiz ──────────────────────────────────────────────
        elif tool_name == "create_quiz":
            topic = args.get("topic", "").strip()
            difficulty = args.get("difficulty", None)

            if not topic:
                return "⚠️ Vui lòng cho biết chủ đề cần tạo đề (ví dụ: FSK, QPSK, BER)."

            # Block student with pending quiz
            pending = self.quiz_storage.get_latest_pending_quiz(student_id)
            if pending:
                user_role = get_user_role(student_id)
                if user_role != "teacher":
                    return (
                        f"❌ Bạn không thể tạo đề mới khi đang có bài chưa nộp!\n\n"
                        f"📋 **Bài kiểm tra chưa hoàn thành:**\n"
                        f"- Chủ đề: {pending.get('topic', 'N/A')}\n\n"
                        "💡 **Bạn có thể:**\n"
                        "1. **Xem lại đề:** Gõ \"xem lại đề\"\n"
                        "2. **Nộp bài:**\n"
                        "```\nNộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B\n```"
                    )

            result = self.quiz_generator.generate_quiz(
                subject="Điều chế số",
                topic=topic,
                difficulty=difficulty,
                use_student_difficulty=(difficulty is None),
            )

            if not result["success"]:
                return f"❌ Không thể tạo đề kiểm tra: {result['error']}\n\n💡 Vui lòng thử lại."

            if not result.get("answer_key"):
                return "❌ Lỗi: Không thể tạo đề vì thiếu đáp án. Vui lòng thử lại."

            try:
                quiz_id = self.quiz_storage.save_quiz(
                    student_id=student_id,
                    content=result["quiz_markdown"],
                    answer_key=result["answer_key"],
                    subject="Điều chế số",
                    topic=topic,
                    difficulty=result["metadata"]["difficulty"],
                )
                print(f"✅ Saved quiz: {quiz_id}")
            except Exception as e:
                print(f"⚠️ Could not save quiz: {e}")

            user_role = get_user_role(student_id)
            submission_note = (
                ""
                if user_role == "teacher"
                else "\n\n💡 **Để nộp bài:**\n```\nNộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B\n```"
            )

            return f"✅ Đã tạo xong đề kiểm tra!\n\n{result['quiz_markdown']}{submission_note}"

        # ── submit_quiz ──────────────────────────────────────────────
        elif tool_name == "submit_quiz":
            answers_text = args.get("answers_text", user_query)
            pending = self.quiz_storage.get_latest_pending_quiz(student_id)

            if not pending:
                return (
                    "❌ Chưa có bài kiểm tra nào được tạo!\n\n"
                    "💡 Bạn có thể tạo đề mới bằng cách nói: \"Tạo đề về FSK\""
                )

            answers = self._extract_answers(answers_text)
            if not answers:
                return (
                    "❌ Không thể đọc được đáp án!\n\n"
                    "💡 **Format đúng:**\n"
                    "```\nNộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B\n```\n"
                    "⚠️ Cần đủ 10 câu, format: số-chữ cái (VD: 1-A, 2-B)"
                )

            quiz = self.quiz_storage.get_quiz(pending["id"])
            if not quiz:
                return f"❌ Lỗi: Không tìm thấy quiz {pending['id']}"

            if self.submission_manager.check_quiz_submitted(pending["id"], student_id):
                return (
                    f"❌ Bài này đã được nộp rồi!\n\n"
                    f"📋 Quiz ID: `{pending['id']}`\n\n"
                    "💡 Bạn có thể tạo đề mới bằng cách nói: \"Tạo đề về QPSK\""
                )

            answer_key = quiz.get("answer_key")
            if not answer_key:
                return "❌ Lỗi: Đề thi thiếu đáp án. Vui lòng liên hệ admin."

            result = self.submission_manager.submit_quiz(
                quiz_id=pending["id"],
                student_id=student_id,
                student_answers=answers,
                answer_key=answer_key,
            )

            if not result["success"]:
                return f"❌ Lỗi nộp bài: {result.get('error', 'Unknown error')}"

            self.quiz_storage.update_quiz_status(pending["id"], "completed")

            # Async daily stats update
            today = datetime.now().strftime("%Y-%m-%d")
            threading.Thread(
                target=self._call_daily_stats,
                args=(student_id, today),
                daemon=True,
            ).start()

            detailed = self.submission_manager.get_submission_with_details(
                result["submission_id"], answer_key
            )

            details_lines = []
            for d in detailed["details"]:
                icon = "✅" if d["is_correct"] else "❌"
                if d["is_correct"]:
                    details_lines.append(f"   {icon} Câu {d['question_number']}: {d['student_answer']} (Đúng)")
                else:
                    details_lines.append(
                        f"   {icon} Câu {d['question_number']}: {d['student_answer']} → Đúng là {d['correct_answer']}"
                    )

            return (
                f"🎉 **ĐÃ NỘP BÀI THÀNH CÔNG!**\n\n"
                f"📊 **KẾT QUẢ:**\n"
                f"- Điểm: **{result['score']}/{result['total']}** ({result['percentage']:.1f}%)\n"
                f"- Đúng: {detailed['correct_count']} câu\n"
                f"- Sai: {detailed['incorrect_count']} câu\n"
                f"- Thời gian: {result['duration']} phút\n\n"
                f"📝 **CHI TIẾT:**\n"
                + "\n".join(details_lines)
                + f"\n\n💾 Lần nộp thứ {result['daily_count']} hôm nay"
            )

        # ── view_quiz ────────────────────────────────────────────────
        elif tool_name == "view_quiz":
            pending = self.quiz_storage.get_latest_pending_quiz(student_id)
            if not pending:
                return "📭 Hiện không có đề kiểm tra nào đang chờ.\n\n💡 Nói \"Tạo đề về FSK\" để bắt đầu."
            return self._show_quiz_content(pending)

        # ── draw_graph ───────────────────────────────────────────────
        elif tool_name == "draw_graph":
            equation = args.get("equation", "")
            x_min = args.get("x_min", -10)
            x_max = args.get("x_max", 10)

            if not equation:
                return "⚠️ Không thể xác định hàm số. Vui lòng nhập rõ hơn (VD: 'vẽ y = x**2')."

            result = self.graph_generator.generate_graph(equation, x_min, x_max)

            if result["success"]:
                return (
                    f"✅ Đã vẽ xong đồ thị!\n\n"
                    f"📊 **Thông tin:**\n"
                    f"- Hàm số: y = {equation}\n"
                    f"- Khoảng: x ∈ [{x_min}, {x_max}]\n"
                    f"- Kích thước: {result['file_size']/1024:.1f}KB\n\n"
                    f"[IMAGE:{result['file_path']}]\n\n"
                    "💡 Bạn có muốn tôi giải thích gì về đồ thị này không?"
                )
            else:
                return (
                    f"❌ Không thể vẽ đồ thị: {result['error']}\n\n"
                    "💡 Gợi ý: Kiểm tra cú pháp hàm số (VD: x**2, np.sin(x), 2*x+3)"
                )

        # ── general_chat ─────────────────────────────────────────────
        elif tool_name == "general_chat":
            return self.rag.generate_casual(args.get("message", user_query))

        return "⚠️ Tool không xác định."

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────
    def _build_system_prompt(self, student_id: str) -> str:
        pending = self.quiz_storage.get_latest_pending_quiz(student_id)
        pending_warning = ""
        if pending:
            pending_warning = (
                f"\n\n⚠️ CẢNH BÁO: Người dùng đang có bài kiểm tra chưa nộp!\n"
                f"- Chủ đề: {pending.get('topic', 'N/A')}\n"
                "Nếu người dùng hỏi câu hỏi kỹ thuật liên quan đến bài đang làm, hãy gọi tool rag_answer và để quiz_guard quyết định.\n"
                "Nếu người dùng muốn tạo đề mới, hãy gọi create_quiz và hệ thống sẽ kiểm tra role."
            )

        return (
            "Bạn là trợ lý AI chuyên về điều chế số (digital modulation & communications).\n\n"
            "Bạn có thể:\n"
            "- Trả lời câu hỏi kỹ thuật từ tài liệu (rag_answer)\n"
            "- Tạo bài kiểm tra trắc nghiệm (create_quiz)\n"
            "- Nhận và chấm bài nộp (submit_quiz)\n"
            "- Hiển thị lại đề đang làm (view_quiz)\n"
            "- Vẽ đồ thị hàm số (draw_graph)\n"
            "- Trò chuyện thông thường (general_chat)\n\n"
            "Hãy chọn đúng tool dựa trên ý định của người dùng."
            + pending_warning
        )

    def _extract_answers(self, text: str) -> Optional[str]:
        """Parse '1-A,2-B,...' from user text. Returns normalized string or None."""
        for kw in ["nộp bài:", "nộp:", "submit:"]:
            text = text.lower().replace(kw, "")

        matches = re.findall(r"(\d+)\s*-?\s*([A-D])", text, re.IGNORECASE)
        if len(matches) < 10:
            print(f"   ⚠️ Found only {len(matches)} answers, need 10")
            return None

        result = ",".join(f"{n}-{l.upper()}" for n, l in matches[:10])
        print(f"   ✓ Extracted answers: {result}")
        return result

    def _show_quiz_content(self, pending: Dict) -> str:
        content = pending.get("content", "")
        if not content:
            return (
                f"⚠️ Không thể tải nội dung đề!\n"
                f"- Quiz ID: `{pending.get('id')}`\n"
                f"- Chủ đề: {pending.get('topic', 'N/A')}"
            )
        return (
            f"📋 **ĐỀ KIỂM TRA ĐANG LÀM**\n\n{content}\n\n"
            "💡 **Để nộp bài:**\n```\nNộp bài: 1-A,2-B,3-C,4-D,5-A,6-B,7-C,8-D,9-A,10-B\n```"
        )

    def _extract_text_from_image(self, image_context: Dict) -> str:
        """OCR via GPT-4o Vision."""
        try:
            r = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Trích xuất TOÀN BỘ nội dung văn bản trong ảnh.\n"
                                "Giữ nguyên format, xuống dòng, ký hiệu đặc biệt.\n"
                                "Chỉ trả về text được trích xuất, không thêm giải thích."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_context['base64']}",
                                "detail": "high",
                            },
                        },
                    ],
                }],
                max_tokens=1500,
                temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"   ⚠️ OCR error: {e}")
            return ""

    def _extract_hidden_text(self, content: str) -> str:
        """Extract text from HTML comment markers in conversation history."""
        content = re.sub(r"!\[.*?\]\(.*?\)\s*", "", content)
        match = re.search(r"<!-- EXTRACTED_TEXT\s+(.*?)\s+-->", content, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            user_text = content.split("<!-- EXTRACTED_TEXT")[0].strip()
            return f"{user_text}\n\n{extracted}" if user_text else extracted
        return content

    def _call_daily_stats(self, student_id: str, date: str):
        """Background call to update daily evaluation stats."""
        try:
            api_base = os.getenv("API_BASE_URL", "http://localhost:8110")
            requests.get(
                f"{api_base}/api/stats/daily",
                params={"student_id": student_id, "date": date},
                timeout=5,
            )
        except Exception as e:
            print(f"⚠️ Daily stats update failed: {e}")


# ================== WRAPPER (kept for app.py compatibility) ==================
class ScienceQASystem:
    def __init__(self, student_id: str = None):
        self.agent = DigitalModulationAgent(student_id=student_id)

    def query(
        self,
        user_query: str,
        conversation_history: List[Dict] = None,
        image_context: Optional[Dict] = None,
    ) -> Dict:
        return self.agent.query(user_query, conversation_history, image_context)
