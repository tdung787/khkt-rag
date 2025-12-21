import re
import json

# ======================
#  HÀM ĐỌC FILE TXT RAW
# ======================
def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ============================================
#  HÀM LÀM SẠCH – BỎ HEADER ẢNH, TOKEN, ======
# ============================================
def clean_text(raw):
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        line_strip = line.strip()
        # Bỏ các dòng meta
        if line_strip.startswith("===="): continue
        if line_strip.startswith("📄"): continue
        if line_strip.startswith("Tokens:"): continue
        if line_strip.startswith("Finish:"): continue
        if line_strip.startswith("CHƯƠNG"): continue
        if line_strip.startswith("CÂU HỎI"): continue
        if line_strip.startswith("Author:"): continue
        if line_strip.startswith("Tác giả:"): continue
        if line_strip.startswith("Xử lý"): continue
        # Bỏ dòng rỗng thừa
        cleaned.append(line_strip)
    
    # Gom lại thành 1 đoạn lớn
    text = "\n".join([l for l in cleaned if l != ""])
    return text

# =============================================
#  TÁCH CÁC KHỐI CÂU "Câu 1." hoặc "Câu 1:" → "Câu 2." ...
# =============================================
def split_questions(text):
    # Hỗ trợ cả dấu chấm (.) và dấu hai chấm (:)
    pattern = r"(Câu\s+\d+[\.:].*?)(?=Câu\s+\d+[\.:]|$)"
    blocks = re.findall(pattern, text, flags=re.S)
    return blocks

# ============================================================
#  TÁCH NỘI DUNG CÂU HỎI + 4 ĐÁP ÁN A/B/C/D
# ============================================================
def parse_question_block(block, subject="Vật lý"):
    # Hỗ trợ cả "Câu 1." và "Câu 1:"
    m = re.match(r"Câu\s+(\d+)[\.:]?\s*(.+)", block, flags=re.S)
    if not m:
        return None
    
    q_number = int(m.group(1))
    remain = m.group(2).strip()
    
    # Tìm vị trí option A đầu tiên (có thể cùng dòng hoặc xuống dòng)
    # Dùng word boundary hoặc whitespace trước A.
    first_option = re.search(r"(?<!\w)A\.\s+", remain)
    if first_option:
        question_text = remain[:first_option.start()].strip()
        options_text = remain[first_option.start():].strip()
    else:
        question_text = remain.strip()
        options_text = ""
    
    # Parse options - dừng khi gặp option tiếp theo (phải có space trước)
    option_pattern = r"(?<!\w)(?P<key>[ABCD])\.\s+(?P<val>(?:(?!(?<!\w)[ABCD]\.\s).)+?)(?=(?<!\w)[ABCD]\.\s|$)"
    matches = re.finditer(option_pattern, options_text, flags=re.S)
    
    options = {}
    used_keys = set()
    available_keys = ['A', 'B', 'C', 'D']
    
    for match in matches:
        key = match.group("key")
        val = match.group("val").strip()
        val = " ".join(val.split())
        
        # Phát hiện option có thể bị cắt content (kết thúc bằng "= A." "= B." etc)
        if re.search(r'[=\s][ABCD]\.$', val):
            print(f"⚠️ Câu {q_number}: Option {key} có thể bị cắt ('{val[-10:] if len(val) > 10 else val}')")
        
        if key in used_keys:
            for new_key in available_keys:
                if new_key not in used_keys:
                    print(f"⚠️ Câu {q_number}: Đáp án trùng '{key}' → tự động đổi thành '{new_key}'")
                    key = new_key
                    break
        
        options[key] = val
        used_keys.add(key)
    
    # Tạo subject_code từ tên môn học
    subject_code = subject.lower().replace(" ", "_").replace("ý", "y").replace("á", "a").replace("ế", "e")
    if subject == "Vật lý":
        subject_code = "vat_ly"
    elif subject == "Hóa học":
        subject_code = "hoa_hoc"
    
    return {
        "id": f"cau_{q_number}_{subject_code}",
        "question": question_text,
        "options": options,
        "correct_answer": "",
        "correct_answer_text": "",
        "explanation": "",
        "subject": subject
    }

# ===========================
#  CHẠY TOÀN BỘ QUY TRÌNH
# ============================
def parse_txt_to_json(input_path, output_path, subject="Vật lý"):
    raw = load_txt(input_path)
    cleaned = clean_text(raw)
    blocks = split_questions(cleaned)
    
    print(f"🔍 Tìm thấy {len(blocks)} khối câu hỏi")
    
    results = []
    missing = []
    
    for block in blocks:
        q = parse_question_block(block, subject)
        if not q:
            continue
        
        # Kiểm tra thiếu option A/B/C/D hoặc option trống
        options = q["options"]
        if len(options) < 4 or any(val.strip() == "" for val in options.values()):
            missing.append(q["id"])
        
        results.append(q)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"📁 Đã xuất JSON: {output_path}")
    print(f"📌 Tổng số câu đọc được: {len(results)}")
    if missing:
        print(f"⚠️ Câu thiếu đáp án: {missing}")
        print(f"❗ Tổng số câu thiếu đáp án: {len(missing)}")

# ============================
#  CHẠY DEMO
# ============================
if __name__ == "__main__":
    parse_txt_to_json(
        input_path="data/input/txt/nhiet_hoc_VL-lop10-Q.txt",
        output_path="Vl.json",
        subject="Vật lý"
    )