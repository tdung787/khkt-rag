"""
Module: VietnameseTextNormalizer (improved with batch processing)
Author: AI Assistant

Thêm tính năng:
- Xử lý cả folder (đọc tất cả file .txt)
- Lưu kết quả vào JSON với format: {"page": số_trang, "content": nội_dung}
- Số trang được lấy từ tên file (ví dụ: bt10_text_5_0.txt -> page = 5)
"""

import re
import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class VietnameseTextNormalizer:
    """Chuẩn hóa văn bản tiếng Việt với regex và (tuỳ chọn) LLM."""

    # Bảng lỗi/OCR phổ biến mở rộng
    DEFAULT_TYPOS = {
        # OCR/simple typos
        'phưong': 'phương', 'đêu': 'đều', 'đuợc': 'được', 'tât': 'tất',
        'môt': 'một', 'thưc': 'thực', 'hiên': 'hiện', 'trươc': 'trước',
        'bât': 'bất', 'ngươi': 'người', 'đươc': 'được', 'thêm': 'thêm',
        'nươc': 'nước', 'dươi': 'dưới', 'trươi': 'trời', 'bươc': 'bước',
        'chuân': 'chuẩn', 'tiên': 'tiến', 'nhân': 'nhận', 'gôm': 'gồm',
        'đươg': 'được', 'lươg': 'lượng', 'chiêu': 'chiều', 'thơi': 'thời',
        'luân': 'luận', 'trãi': 'trải', 'trãi nghiệm': 'trải nghiệm',
        'toàn cẩu': 'toàn cầu', 'sản suất': 'sản xuất', 'dây chuyển': 'dây chuyền',
        'sản suất': 'sản xuất', 'quan sát, trãi nghiệm': 'quan sát, trải nghiệm',
        # common short forms
        '\bv\.?v\.?\b': 'v.v.', '\bvv\b': 'v.v.',
        # hyphenation issues
        'Ga-li-lê': 'Ga-li-lê',
    }

    # Các pattern riêng để xử lý (không nên đưa vào DEFAULT_TYPOS nếu cần regex)
    EXTRA_REPLACEMENTS = [
        # sửa spacing và dấu câu thường gặp
        (r'\s*,\s*', ', '),
        (r'\s*\.\s*', '. '),
        (r'\s*;\s*', '; '),
        (r'\s*:\s*', ': '),
        (r'\bA_\b', 'A.'),
        # SỬA: Chỉ thay A/B/C/D -> A./B./C./D. khi Ở ĐẦU DÒNG
        (r'^([A-D])\s+', r'\1. '),  # Thêm ^ vào đây
        # sửa từ ghép hay chữ thường sau marker
        (r'^(?:[A-D]\.|\d+\.)\s*([a-zàáâãèéêìíòóôõùúưăạảấầẩẫậắằẳẵặẹẻẽềềễệỉịọỏốồổỗộớờởỡợũựỳỵỷ])',
        lambda m: m.group(0)[0:-len(m.group(1))] + m.group(1).upper()),
    ]

    def __init__(self, use_llm: bool = False, openai_api_key: Optional[str] = None,
                 custom_typos: Optional[Dict[str, str]] = None):
        self.use_llm = use_llm
        self.typos = self.DEFAULT_TYPOS.copy()
        if custom_typos:
            self.typos.update(custom_typos)

        if use_llm:
            key = openai_api_key or os.getenv('OPENAI_API_KEY')
            if not key:
                raise ValueError("Cần OPENAI_API_KEY")
            self.client = OpenAI(api_key=key)

    def _extract_page_number(self, filename: str) -> int:
        """
        Trích xuất số trang từ tên file.
        Ví dụ: 'page_021.png' -> 21
            'page_001.png' -> 1
            'bt10_text_5_0.txt' -> 5 (fallback cho format cũ)
        """
        # Pattern tìm số sau 'page_'
        match = re.search(r'page_(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Fallback: Pattern tìm số sau '_text_' hoặc '_page_'
        match = re.search(r'_(?:text|page)_(\d+)', filename)
        if match:
            return int(match.group(1))
        
        # Fallback cuối: tìm số đầu tiên trong tên file
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        
        return 0  # Mặc định nếu không tìm thấy

    def _merge_broken_lines(self, lines: list) -> list:
        cleaned_lines = [line.replace('_', '').strip() for line in lines if line.strip()]
        merged = []
        i = 0
        while i < len(cleaned_lines):
            cur = cleaned_lines[i].strip()
            if not cur:
                i += 1
                continue
            if i + 1 < len(cleaned_lines):
                nxt = cleaned_lines[i + 1].strip()
                is_next_marker = (
                    re.match(r'^[A-D](?:[\.\s]|$)', nxt) or 
                    re.match(r'^\d+\.\s+\S', nxt)
                )
                ends_with_punct = re.search(r'[\.\?!:;]$', cur)
                starts_lowercase = nxt and nxt[0].islower()
                is_number_continuation = re.search(r'-$', cur) and re.match(r'^\d+\.$', nxt)
                if (not is_next_marker and (not ends_with_punct or starts_lowercase)) or is_number_continuation:
                    combined = cur + ' ' + nxt
                    merged.append(combined)
                    i += 2
                    continue
            merged.append(cur)
            i += 1
        return merged

    def _apply_typos(self, line: str) -> str:
        # áp dụng các thay thế đơn giản (bao gồm cả regex keys trong typos)
        for typo, correct in self.typos.items():
            try:
                # nếu typo là pattern regex (chứa \b hoặc ký tự đặc biệt), dùng re.sub
                if re.search(r'[^\w\s]', typo) or '\\b' in typo or '(' in typo:
                    line = re.sub(typo, correct, line, flags=re.IGNORECASE)
                else:
                    pattern = re.compile(r'\b' + re.escape(typo) + r'\b', re.IGNORECASE)
                    def repl(m):
                        text = m.group(0)
                        # giữ nguyên kiểu chữ đầu
                        if text[0].isupper():
                            return correct.capitalize()
                        else:
                            return correct
                    line = pattern.sub(repl, line)
            except re.error:
                # fallback: literal replace
                line = line.replace(typo, correct)
        return line

    def _postprocess_punctuation(self, line: str) -> str:
        line = re.sub(r'\s+', ' ', line).strip()
        line = re.sub(r'\s*-\s*', ' - ', line)
        line = re.sub(r'\s+([,;:?!])', r'\1', line)
        line = re.sub(r'([,;:?!])(\S)', r'\1 \2', line)
        line = re.sub(r'([\.\?\!]){2,}', r'\1', line)
        line = re.sub(r'^([A-D])\s+', r'\1. ', line)
        if (
            not re.search(r'[\.\?!:;-]$', line) and
            len(line.split()) > 2 and
            not re.match(r'^[A-D]\.\s*\(\d+\)', line) and
            not re.match(r'^[A-D]\.$', line)
        ):
            line = line + '.'
        return line

    def normalize_with_regex(self, text: str) -> str:
        # Bước 1: Tách dòng khi gặp dấu kết câu + đáp án (A–D)
        raw_text = re.sub(
            r'(?<=[.!?])\s+(?=[A-Da-d]\.)',
            '\n',
            text
        )

        # Bước 2: Chuẩn hóa các dòng thô
        raw_lines = raw_text.split('\n')
        normalized_lines = []

        for ln in raw_lines:
            ln = ln.strip()
            if not ln:
                continue

            # Viết hoa chữ cái đầu tiên nếu là a-d.
            ln = re.sub(r'^([a-d])(\.)', lambda m: f"{m.group(1).upper()}{m.group(2)}", ln)

            normalized_lines.append(ln)

        # Bước 3: Gộp dòng sau khi đã chuẩn hóa chữ cái đầu
        merged = self._merge_broken_lines(normalized_lines)

        # Bước 4: Làm sạch, thay thế lỗi, giữ nguyên viết hoa/thường gốc
        result_lines = []
        for ln in merged:
            ln = ln.strip()
            if not ln:
                continue

            # Bỏ gạch dưới, thừa khoảng trắng
            ln = re.sub(r'_+', '', ln)
            ln = re.sub(r'\s{2,}', ' ', ln)

            # Sửa lỗi OCR
            ln = self._apply_typos(ln)

            # Đảm bảo format A. (không A . hay A_)
            ln = re.sub(r'^([A-D])\s*[_\.]?\s*', r'\1. ', ln)
            ln = re.sub(r'^(\d+\.)\s*', r'\1 ', ln)

            # Thay thế bổ sung
            for pat, repl in self.EXTRA_REPLACEMENTS:
                if callable(repl):
                    ln = re.sub(pat, repl, ln)
                else:
                    ln = re.sub(pat, repl, ln)

            # Giữ nguyên chữ thường sau dấu chấm
            ln = self._postprocess_punctuation(ln)
            ln = re.sub(r'v\.v\.\.', 'v.v.', ln)

            result_lines.append(ln)

        return "\n".join(result_lines)


    def normalize_with_llm(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                    {"role": "system", "content": "Chuẩn hóa văn bản tiếng Việt (sửa lỗi chính tả, dấu câu, khoảng trắng). Giữ nguyên cấu trúc và chỉ trả văn bản đã sửa."},
                    {"role": "user", "content": text}
                    
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Lỗi LLM: {e}")
            return text

        except Exception as e:
            print(f"Lỗi LLM: {e}")
            return text

    def normalize(self, text: str, method: str = 'regex') -> str:
        if method == 'regex':
            return self.normalize_with_regex(text)
        elif method == 'llm':
            if not self.use_llm:
                raise ValueError("Chưa bật use_llm=True")
            return self.normalize_with_llm(text)
        elif method == 'hybrid':
            text = self.normalize_with_regex(text)
            if self.use_llm:
                text = self.normalize_with_llm(text)
            return text
        else:
            raise ValueError(f"Method không hợp lệ: {method}")

    def normalize_file(self, input_path: str, output_path: str, method: str = 'regex'):
        """Chuẩn hóa 1 file đơn lẻ"""
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        result = self.normalize(text, method=method)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"✓ Đã lưu: {output_path}")

    def normalize_folder(self, input_folder: str, output_json: str, output_txt_folder: Optional[str] = None, method: str = 'regex'):
        """
        Chuẩn hóa tất cả file .txt trong folder và lưu vào JSON + các file txt riêng lẻ.
        
        Args:
            input_folder: Đường dẫn đến folder chứa các file .txt
            output_json: Đường dẫn file JSON output
            output_txt_folder: Đường dẫn folder để lưu các file txt đã chuẩn hóa (tùy chọn)
            method: Phương pháp chuẩn hóa ('regex', 'llm', hoặc 'hybrid')
        
        Output:
            - JSON file với format: [{"page": 1, "content": "..."}, ...]
            - Folder chứa các file .txt riêng lẻ (nếu output_txt_folder được chỉ định)
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy folder: {input_folder}")
        
        # Tạo folder output cho txt files nếu được chỉ định
        if output_txt_folder:
            output_txt_path = Path(output_txt_folder)
            output_txt_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Tạo folder output: {output_txt_folder}")
        
        # Lấy tất cả file .txt
        txt_files = sorted(input_path.glob("*.txt"))
        if not txt_files:
            print(f"⚠️ Không tìm thấy file .txt nào trong {input_folder}")
            return
        
        print(f"📂 Tìm thấy {len(txt_files)} file .txt")
        
        results = []
        
        for txt_file in txt_files:
            try:
                # Đọc nội dung
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Chuẩn hóa
                normalized_text = self.normalize(text, method=method)
                
                # Trích xuất số trang từ tên file
                page_num = self._extract_page_number(txt_file.name)
                
                # Thêm vào kết quả JSON
                results.append({
                    "page": page_num,
                    "content": normalized_text,
                    "source_file": txt_file.name
                })
                
                # Lưu file txt riêng lẻ (nếu được yêu cầu)
                if output_txt_folder:
                    output_file = output_txt_path / txt_file.name
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(normalized_text)
                    print(f"✓ Xử lý: {txt_file.name} (trang {page_num}) → {output_file.name}")
                else:
                    print(f"✓ Xử lý: {txt_file.name} (trang {page_num})")
                
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {txt_file.name}: {e}")
        
        # Sắp xếp theo số trang
        results.sort(key=lambda x: x['page'])
        
        # Lưu vào JSON
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Hoàn tất!")
        print(f"   - JSON: {output_json} ({len(results)} trang)")
        if output_txt_folder:
            print(f"   - TXT files: {output_txt_folder} ({len(txt_files)} files)")
        
        return output_json
