import json
import re

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_answer_key(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    answer_map = {}

    pattern = r"Câu\s+(\d+)\s*[:\-]\s*([ABCD])"

    for line in lines:
        m = re.search(pattern, line.strip(), flags=re.IGNORECASE)
        if m:
            q_num = int(m.group(1))
            ans = m.group(2).upper()
            answer_map[q_num] = ans

    return answer_map


def map_answers(json_data, answer_key):
    missing = []

    for item in json_data:
        # Lấy số câu từ id: "cau_274"
        m = re.search(r"(\d+)", item["id"])
        if not m:
            missing.append(item["id"])
            continue

        q_num = int(m.group(1))

        if q_num not in answer_key:
            missing.append(item["id"])
            continue

        correct_letter = answer_key[q_num]
        item["correct_answer"] = correct_letter

        # Lấy chính nội dung của đáp án
        if "options" in item and correct_letter in item["options"]:
            item["correct_answer_text"] = item["options"][correct_letter]
        else:
            item["correct_answer_text"] = None
            missing.append(item["id"])

    return json_data, missing


def save_json(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_json = "data/input/json/nhiet_hoc_VL-lop10-Q.json"       # đổi theo tên file của bạn
    answer_txt = "data/input/txt/nhiet_hoc_VL-lop10-A.txt"      # đổi theo tên file của bạn
    output_json = "output_mapped.json"

    print("📥 Đọc file JSON…")
    data = load_json(input_json)

    print("📥 Đọc file đáp án…")
    answer_key = load_answer_key(answer_txt)

    print("🔗 Mapping đáp án…")
    mapped, missing = map_answers(data, answer_key)

    print(f"💾 Lưu file: {output_json}")
    save_json(mapped, output_json)

    if missing:
        print("\n⚠️ Các câu KHÔNG tìm thấy đáp án:")
        for mid in missing:
            print("  -", mid)
    else:
        print("\n✅ Tất cả câu đều được gán đáp án!")
