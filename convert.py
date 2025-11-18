import json

# Đường dẫn file
input_file = "data/input/json/nhiet_hoc_VL-lop10-E.json" 
output_file = "output.json"  # file kết quả

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Đổi id cho từng câu hỏi
for item in data:
    old_id = item["id"]              # ví dụ: "cau_25"
    new_id = old_id + "_vat_ly"      # → "cau_25_vat_ly"
    item["id"] = new_id

# Ghi lại ra file mới (đẹp, có thứ tự như cũ)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Đã xong! File mới lưu tại: {output_file}")