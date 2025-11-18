import os
import textwrap
import traceback
from openai import OpenAI
import tempfile
import subprocess

class PhysicsSimulator:
    def __init__(self, client: OpenAI):
        self.client = client
        self.output_path = "output_physics.png"

    def simulate(self, user_query: str):
        """
        Tự động sinh code mô phỏng vật lý từ mô tả người dùng
        """
        prompt = f"""
Bạn là lập trình viên vật lý học. Viết mã Python mô phỏng hiện tượng sau bằng matplotlib:
'{user_query}'

Yêu cầu:
- Lưu kết quả mô phỏng hoặc đồ thị thành ảnh PNG tên 'output_physics.png'
- Không yêu cầu input() hoặc print()
- Không dùng animation hay interactive, chỉ cần 1 ảnh tĩnh
- Mã phải tự chạy được ngay khi gọi exec()

Ví dụ:
Nếu người dùng nói 'mô phỏng rơi tự do có lực cản', bạn có thể mô phỏng quãng đường và vận tốc theo thời gian.
"""
        try:
            # Gọi LLM sinh code mô phỏng
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
            )
            code = response.choices[0].message.content.strip()

            # Trích mã từ code block
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]

            # Ghi tạm vào file
            temp_file = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
            with open(temp_file.name, "w") as f:
                f.write(code)

            # Chạy code an toàn (tách process)
            subprocess.run(["python", temp_file.name], timeout=15, check=True)

            # Kiểm tra file ảnh
            if os.path.exists(self.output_path):
                size = os.path.getsize(self.output_path) / 1024
                return {
                    "success": True,
                    "file_path": self.output_path,
                    "file_size": size,
                    "code": code,
                }
            else:
                return {"success": False, "error": "Không tạo được file ảnh."}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Quá thời gian chạy code."}
        except Exception as e:
            return {"success": False, "error": traceback.format_exc()}
