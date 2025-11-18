# src/utils/filter_header_footer.py
import os
import re
from pathlib import Path

def remove_headers_and_footers(input_folder: str, output_folder: str):
    """
    Lọc bỏ header và footer khỏi tất cả file .txt trong folder đầu vào.
    Header: dòng chứa 'Cô Nhung Cute' hoặc 'VẬT LÍ'
    Footer: dòng chỉ có số trang (chỉ chứa số)
    Giữ nguyên toàn bộ nội dung khác.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in input_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        filtered_lines = []
        skip_next = False

        for line in lines:
            stripped = line.strip()

            # Loại header
            if re.search(r"Cô\s+Nhung\s+Cute", stripped) or "VẬT LÍ" in stripped:
                skip_next = True  # bỏ luôn dòng kế tiếp (thường là "10")
                continue

            # Nếu dòng sau header (thường là "10")
            if skip_next:
                skip_next = False
                continue

            # Loại footer: chỉ có số
            if re.fullmatch(r"\d+", stripped):
                continue

            filtered_lines.append(line)

        # Ghi file đã lọc ra output
        out_file = output_path / file_path.name
        with open(out_file, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)

        print(f"✅ Đã lọc xong: {file_path.name}")

    print("\n🎯 Hoàn tất lọc tất cả file!")