import re
import json

def load_json(path):
    """Đọc file JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    """Lưu file JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_explanations(txt_path):
    """
    Parse file lời giải - HỖ TRỢ MỌI FORMAT
    
    Return: {số_câu: "lời_giải"}
    """
    print(f"\n📥 Đọc file: {txt_path}")
    
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Clean meta lines
    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
        line_strip = line.strip()
        # Bỏ header, footer, meta
        if any(line_strip.startswith(x) for x in ["=====", "📄", "Tokens:", "Finish:"]):
            continue
        cleaned_lines.append(line)
    
    content = "\n".join(cleaned_lines)
    
    print(f"📊 Đã clean: {len(content):,} ký tự")
    
    # TẤT CẢ PATTERNS - Không return sớm, gộp tất cả
    all_explanations = {}
    
    # ============================================
    # PATTERN 1: Có "Hướng dẫn giải"
    # ============================================
    pattern1 = r"Câu\s+(\d+)[\.:]?\s*[A-D]?\s*\nHướng dẫn giải\s*(.*?)(?=\nCâu\s+\d+|\Z)"
    matches1 = re.findall(pattern1, content, flags=re.S | re.IGNORECASE)
    
    for qnum, exp in matches1:
        all_explanations[int(qnum)] = exp.strip()
    
    if matches1:
        print(f"✓ Pattern 1 (Hướng dẫn giải): {len(matches1)} câu")
    
    # ============================================
    # PATTERN 2: Có đáp án rõ ràng (A/B/C/D)
    # ============================================
    pattern2 = r"Câu\s+(\d+)[\.:\s→\-]*(?:Chọn\s+đáp\s+án|Đáp\s+án|đáp\s+án)?\s*([A-D])\s*\n(.*?)(?=\nCâu\s+\d+|\Z)"
    matches2 = re.findall(pattern2, content, flags=re.S | re.IGNORECASE)
    
    for qnum, answer, exp in matches2:
        qnum_int = int(qnum)
        # Chỉ thêm nếu chưa có (Pattern 1 ưu tiên)
        if qnum_int not in all_explanations:
            all_explanations[qnum_int] = exp.strip()
    
    if matches2:
        new_from_p2 = len([q for q, _, _ in matches2 if int(q) not in all_explanations or all_explanations.get(int(q)) == ""])
        print(f"✓ Pattern 2 (có đáp án A/B/C/D): {len(matches2)} câu (thêm {new_from_p2} mới)")
    
    # ============================================
    # PATTERN 3: Không có đáp án - chỉ giải thích
    # ============================================
    pattern3 = r"Câu\s+(\d+)[\.:]?\s*\n(.*?)(?=\nCâu\s+\d+|\Z)"
    matches3 = re.findall(pattern3, content, flags=re.S)
    
    for qnum, exp in matches3:
        qnum_int = int(qnum)
        # Chỉ thêm nếu chưa có
        if qnum_int not in all_explanations:
            # Bỏ dòng đầu nếu chỉ là đáp án đơn (A, B, C, D)
            exp_lines = exp.strip().split("\n")
            if len(exp_lines) > 0:
                first_line = exp_lines[0].strip()
                if re.match(r'^(?:Đáp\s+án\s+)?[A-D]\.?\s*$', first_line, re.IGNORECASE):
                    exp = "\n".join(exp_lines[1:])
            
            all_explanations[qnum_int] = exp.strip()
    
    new_from_p3 = len([q for q, _ in matches3 if int(q) not in [int(x) for x, _, _ in matches2] and int(q) not in [int(x) for x, _ in matches1]])
    if new_from_p3 > 0:
        print(f"✓ Pattern 3 (không đáp án): Thêm {new_from_p3} câu mới")
    
    # Tổng kết
    print(f"\n📊 Tổng kết:")
    print(f"   - Tổng số câu parse được: {len(all_explanations)}")
    
    if all_explanations:
        sorted_nums = sorted(all_explanations.keys())
        print(f"   - Câu đầu: {sorted_nums[0]}")
        print(f"   - Câu cuối: {sorted_nums[-1]}")
        print(f"   - Danh sách: {sorted_nums[:10]}" + ("..." if len(sorted_nums) > 10 else ""))
    else:
        print("   ❌ KHÔNG PARSE ĐƯỢC CÂU NÀO!")
        print(f"   Preview 300 ký tự đầu:")
        print(content[:300])
    
    return all_explanations


def map_explanations(json_data, explanation_map):
    """
    Gán lời giải vào JSON
    
    Return: (json_data_updated, mapped_list, missing_list)
    """
    print(f"\n🔗 Bắt đầu gán lời giải...")
    print(f"   - JSON có: {len(json_data)} câu")
    print(f"   - Lời giải có: {len(explanation_map)} câu")
    
    # Build map: số câu -> index trong JSON
    id_map = {}
    
    for idx, item in enumerate(json_data):
        # Tìm số trong id
        m = re.search(r'(\d+)', item["id"])
        if m:
            qnum = int(m.group(1))
            id_map[qnum] = idx
    
    print(f"   - JSON parse được: {len(id_map)} số câu")
    
    if id_map:
        sorted_json_nums = sorted(id_map.keys())
        print(f"   - JSON có câu: {sorted_json_nums[:5]}" + ("..." if len(sorted_json_nums) > 5 else ""))
    
    if explanation_map:
        sorted_exp_nums = sorted(explanation_map.keys())
        print(f"   - Lời giải có câu: {sorted_exp_nums[:5]}" + ("..." if len(sorted_exp_nums) > 5 else ""))
    
    # Tìm câu khớp
    common = set(id_map.keys()) & set(explanation_map.keys())
    print(f"\n✓ Số câu KHỚP (có trong cả 2): {len(common)}")
    
    # Gán
    mapped = []
    missing_in_json = []
    
    for qnum in sorted(explanation_map.keys()):
        if qnum in id_map:
            idx = id_map[qnum]
            json_data[idx]["explanation"] = explanation_map[qnum]
            mapped.append(qnum)
        else:
            missing_in_json.append(qnum)
    
    print(f"\n✅ Đã gán: {len(mapped)} câu")
    if mapped:
        print(f"   → Câu: {mapped[:10]}" + ("..." if len(mapped) > 10 else ""))
    
    if missing_in_json:
        print(f"\n⚠️  Không gán được (không có trong JSON): {len(missing_in_json)} câu")
        print(f"   → Câu: {missing_in_json[:10]}" + ("..." if len(missing_in_json) > 10 else ""))
    
    return json_data, mapped, missing_in_json


if __name__ == "__main__":
    # ==============================
    # CONFIG - SỬA Ở ĐÂY
    # ==============================
    input_json = "data/input/json/ester-lipid_hoa12_A.json"
    explanation_txt = "data/input/txt/ester-lipid_hoa12-E.txt"
    output_json = "data/input/json/ester-lipid_hoa12_E.json"
    
    print("="*80)
    print("🔄 CONVERT A → E (GÁN LỜI GIẢI)")
    print("="*80)
    
    # 1. Đọc JSON
    print(f"\n📋 Đọc JSON: {input_json}")
    data = load_json(input_json)
    print(f"✓ Đã đọc {len(data)} câu hỏi")
    
    # 2. Parse lời giải
    explanation_map = parse_explanations(explanation_txt)
    
    if not explanation_map:
        print("\n❌ KHÔNG PARSE ĐƯỢC LỜI GIẢI NÀO!")
        print("💡 Kiểm tra lại file lời giải")
        exit(1)
    
    # 3. Gán vào JSON
    new_data, mapped, missing = map_explanations(data, explanation_map)
    
    # 4. Lưu file
    print(f"\n💾 Lưu file: {output_json}")
    save_json(new_data, output_json)
    print(f"✓ Đã lưu thành công!")
    
    # 5. Tổng kết
    print("\n" + "="*80)
    print("✅ HOÀN TẤT")
    print("="*80)
    print(f"📊 Kết quả:")
    print(f"   - Tổng câu trong JSON: {len(data)}")
    print(f"   - Đã gán lời giải: {len(mapped)}")
    print(f"   - Không gán được: {len(missing)}")
    
    if len(mapped) == len(data):
        print(f"\n🎉 HOÀN HẢO! Tất cả câu đều có lời giải!")
    elif len(mapped) > 0:
        print(f"\n✓ OK! {len(mapped)}/{len(data)} câu có lời giải")
        if missing:
            print(f"   (Còn {len(missing)} câu thiếu lời giải)")
    else:
        print(f"\n⚠️ KHÔNG GÁN ĐƯỢC CÂU NÀO!")
        print(f"💡 Kiểm tra xem số câu có khớp không:")
        print(f"   - JSON có: {[item['id'] for item in data[:3]]}")
        print(f"   - TXT có: {list(explanation_map.keys())[:3]}")