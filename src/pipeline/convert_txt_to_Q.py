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
        cleaned.append(line_strip)
    
    text = "\n".join([l for l in cleaned if l != ""])
    return text

# =============================================
#  TÁCH CÁC KHỐI CÂU
# =============================================
def split_questions(text):
    pattern = r"(Câu\s+\d+[\.:].*?)(?=Câu\s+\d+[\.:]|$)"
    blocks = re.findall(pattern, text, flags=re.S)
    return blocks

# ============================================================
#  LỌC 1: KIỂM TRA BẢNG ĐÁP ÁN
# ============================================================
def is_answer_table(block):
    """
    Bảng đáp án có:
    - Nhiều "Câu N" liên tiếp (>5)
    - Chỉ có chữ cái đơn A/B/C/D (không có "A. nội dung dài")
    """
    cau_matches = re.findall(r'Câu\s+\d+', block)
    
    if len(cau_matches) > 5:
        # Đếm chữ cái đơn (A, B, C, D không có dấu chấm sau)
        single_letters = re.findall(r'\b[A-D]\b(?!\.)', block)
        
        if len(single_letters) >= len(cau_matches) * 0.7:
            return True
    
    # Kiểm tra có nhiều số 3 chữ số không (261, 262, 263...)
    numbers = re.findall(r'\b\d{3}\b', block)
    if len(numbers) > 10:
        return True
    
    return False

# ============================================================
#  LỌC 2: KIỂM TRA LỜI GIẢI
# ============================================================
def is_explanation_block(block):
    """
    Lời giải có:
    - Từ khóa: "Chọn đáp án", "sai vì", "Hướng dẫn giải"...
    - Ký hiệu (a), (b), (c)
    """
    keywords = [
        "Chọn đáp án", "chọn đáp án",
        "Hướng dẫn giải", "hướng dẫn giải",
        "sai vì", "đúng vì",
        "→ Chọn", "Do đó chọn"
    ]
    
    for kw in keywords:
        if kw in block:
            return True
    
    # Kiểm tra (a), (b), (c) - đặc trưng lời giải
    parenthesis = re.findall(r'\([a-d]\)', block.lower())
    if len(parenthesis) >= 2:
        return True
    
    return False

# ============================================================
#  LỌC 3: KIỂM TRA CÂU HỎI HỢP LỆ
# ============================================================
def is_valid_question_block(block):
    """
    Câu hỏi hợp lệ phải có đủ 4 options: A. B. C. D.
    """
    options_with_dot = re.findall(r'\b([A-D])\.\s+\S', block)
    unique_options = set(options_with_dot)
    
    return len(unique_options) >= 4

# ============================================================
#  PARSE CÂU HỎI
# ============================================================
def parse_question_block(block, subject="Vật lý"):
    m = re.match(r"Câu\s+(\d+)[\.:]?\s*(.+)", block, flags=re.S)
    if not m:
        return None
    
    q_number = int(m.group(1))
    remain = m.group(2).strip()
    
    # Tìm option A
    first_option = re.search(r"(?<!\w)A\.\s+", remain)
    if first_option:
        question_text = remain[:first_option.start()].strip()
        options_text = remain[first_option.start():].strip()
    else:
        return None
    
    # Parse options
    option_pattern = r"(?<!\w)(?P<key>[ABCD])\.\s+(?P<val>(?:(?!(?<!\w)[ABCD]\.\s).)+?)(?=(?<!\w)[ABCD]\.\s|$)"
    matches = re.finditer(option_pattern, options_text, flags=re.S)
    
    options = {}
    used_keys = set()
    
    for match in matches:
        key = match.group("key")
        val = match.group("val").strip()
        val = " ".join(val.split())
        
        if key not in used_keys:
            options[key] = val
            used_keys.add(key)
    
    if len(options) < 4:
        return None
    
    # Subject code
    subject_code = subject.lower().replace(" ", "_")
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
#  MAIN FUNCTION
# ============================
def parse_txt_to_json(input_path, output_path, subject="Vật lý"):
    raw = load_txt(input_path)
    cleaned = clean_text(raw)
    blocks = split_questions(cleaned)
    
    print(f"🔍 Tìm thấy {len(blocks)} khối")
    
    results = []
    stats = {
        'total': len(blocks),
        'answer_tables': 0,
        'explanations': 0,
        'invalid': 0,
        'valid': 0
    }
    
    for block in blocks:
        # LỌC 1: Bảng đáp án
        if is_answer_table(block):
            stats['answer_tables'] += 1
            continue
        
        # LỌC 2: Lời giải
        if is_explanation_block(block):
            stats['explanations'] += 1
            continue
        
        # LỌC 3: Không hợp lệ
        if not is_valid_question_block(block):
            stats['invalid'] += 1
            continue
        
        # Parse
        q = parse_question_block(block, subject)
        if not q:
            stats['invalid'] += 1
            continue
        
        # Check options không rỗng
        if any(val.strip() == "" for val in q["options"].values()):
            stats['invalid'] += 1
            continue
        
        stats['valid'] += 1
        results.append(q)
    
    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Stats
    print(f"\n{'='*60}")
    print(f"📊 THỐNG KÊ LỌC:")
    print(f"{'='*60}")
    print(f"   Tổng khối:                {stats['total']}")
    print(f"   ├─ Bảng đáp án (bỏ):     {stats['answer_tables']}")
    print(f"   ├─ Lời giải (bỏ):        {stats['explanations']}")
    print(f"   ├─ Không hợp lệ (bỏ):    {stats['invalid']}")
    print(f"   └─ ✅ CÂU HỎI HỢP LỆ:    {stats['valid']}")
    print(f"{'='*60}")
    print(f"📁 Đã xuất JSON: {output_path}")
    print(f"📌 Tổng số câu đọc được: {len(results)}")

if __name__ == "__main__":
    parse_txt_to_json(
        input_path="data/input/txt/nhiet_hoc_VL-lop10-Q.txt",
        output_path="Vl.json",
        subject="Vật lý"
    )