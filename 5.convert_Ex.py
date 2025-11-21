import re
import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_explanations(txt_path):
    """
    Trích xuất các block:
    Câu X. A
    Hướng dẫn giải
    <đoạn giải thích>

    Return: { X: "<giải thích>" }
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"Câu\s+(\d+)\.\s*[A-D]\s*\nHướng dẫn giải\s*(.*?)(?=\nCâu\s+\d+\.|\Z)"
    matches = re.findall(pattern, content, flags=re.S)

    explanation_map = {}

    for qnum, exp in matches:
        explanation_map[int(qnum)] = exp.strip()

    return explanation_map


def map_explanations(json_data, explanation_map):
    """
    Gán explanation dựa trên explanation_map.
    Chỉ log theo CÁC CÂU xuất hiện trong explanation_map.
    """

    id_map = {}  # map số câu -> index JSON

    for idx, item in enumerate(json_data):
        m = re.search(r"(\d+)", item["id"])
        if not m:
            continue
        id_map[int(m.group(1))] = idx

    mapped = []
    missing = []

    for qnum, explanation in explanation_map.items():
        if qnum in id_map:
            json_data[id_map[qnum]]["explanation"] = explanation
            mapped.append(qnum)
        else:
            missing.append(qnum)

    return json_data, mapped, missing


if __name__ == "__main__":
    input_json = "data/input/json/nhiet_hoc_VL-lop10-A.json"
    explanation_txt = "data/input/txt/nhiet_hoc_VL-lop10-E.txt"
    output_json = "output_with_explanations.json"

    print("📥 Đọc JSON…")
    data = load_json(input_json)

    print("📥 Đọc TXT…")
    explanation_map = parse_explanations(explanation_txt)

    print("\n📊 Tổng số câu tìm thấy trong TXT:", len(explanation_map))

    print("🧩 Đang gán explanation…")
    new_data, mapped, missing = map_explanations(data, explanation_map)

    print(f"\n✅ Gán thành công {len(mapped)}/{len(explanation_map)} câu")
    print("   → Các câu đã gán:", mapped)

    if missing:
        print("\n⚠️ KHÔNG tìm thấy trong JSON (chỉ dựa vào TXT):")
        for q in missing:
            print("  - Câu", q)

    save_json(new_data, output_json)
    print("\n💾 Đã lưu JSON:", output_json)
