import os
from pathlib import Path
import json
import base64
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_to_base64(image_path):
    """Chuyển ảnh thành base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ocr_with_openai(image_path, detail="low", model="gpt-4o"):
    """ OCR sử dụng OpenAI Vision API - ĐÃ KHẮC PHỤC VẤN ĐỀ BỊ CẮT """
    try:
        base64_image = encode_image_to_base64(image_path)

        # PROMPT TỐI ƯU - Ngắn gọn, rõ ràng
        prompt = """
YÊU CẦU ĐỊNH DẠNG:
- Giữ nguyên định dạng gốc (xuống dòng, thụt lề, dấu tiếng Việt)
- Bao gồm mọi ký hiệu, số, chữ cái, công thức
- Không bỏ sót bất kỳ phần nào
- Tuyệt đối không được bao văn bản trong dấu ``` hoặc bất kỳ dạng markdown code block nào
QUY TẮC CHO CÂU TRẮC NGHIỆM:
- Nếu câu hỏi có các đáp án A, B, C, D thì bắt buộc:
+ Mỗi đáp án phải xuống dòng riêng
+ Không được để A, B, C, D nằm trên cùng một dòng
Ví dụ đúng:
A. (1) và (2)
B. (2) và (3)
C. (3) và (1)
D. cả (1), (2) và (3)
YÊU CẦU ĐẶC BIỆT CHO HÓA – SINH – TOÁN:
- Nhận dạng đúng chỉ số dưới (H₂O → H_2O), số mũ (Na⁺ → Na^+), mũ hóa trị, chỉ số phân tử
- Nhận đúng mũ và chỉ số của phương trình hóa học, phương trình toán học, công thức vật lý
- Giữ nguyên mũi tên phản ứng (→, ↔, ⇌, ↓, ↑)
- Nếu OCR nhầm chi → chỉ cần phục hồi đúng từ ngữ dựa trên ngữ cảnh
CHỈ TRẢ VỀ VĂN BẢN OCR:
- Không thêm lời giải thích
- Không thêm ghi chú
- Không được chèn vào code block
INPUT:
(Phần văn bản OCR từ ảnh)
OUTPUT:
Chỉ trả về văn bản đã chuẩn hóa theo đúng định dạng trên
"""


        # Tự động chọn max_tokens dựa trên model
        if model == "gpt-4o-mini":
            max_tokens = 16000
        else:
            max_tokens = 16000

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": detail
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
        )

        extracted_text = response.choices[0].message.content
        usage = response.usage
        finish_reason = response.choices[0].finish_reason

        # CẢNH BÁO NẾU BỊ CẮT
        if finish_reason == 'length':
            print(f" ⚠️ CẢNH BÁO: Output bị cắt do vượt max_tokens!")
            print(f" → Văn bản có thể THIẾU! Cân nhắc tăng max_tokens hoặc dùng gpt-4o")

        return {
            'success': True,
            'text': extracted_text,
            'model': model,
            'detail': detail,
            'usage': {
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens
            },
            'finish_reason': finish_reason,
            'truncated': finish_reason == 'length'
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'text': ''
        }

def process_exam_with_openai(folder_path, detail="low", model="gpt-4o", delay=0.5, output_name='openai_ocr_results'):
    """Xử lý hàng loạt ảnh với OpenAI Vision API"""
    print("🚀 BẮT ĐẦU XỬ LÝ VỚI OPENAI VISION API")
    print(f" Model: {model}")
    print(f" Detail: {detail}")
    print("="*80)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    folder = Path(folder_path)
    image_files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])

    print(f"\n📚 Tìm thấy {len(image_files)} ảnh\n")

    results = {}
    total_tokens = 0
    success_count = 0
    truncated_count = 0

    for idx, image_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 📄 {image_file.name}")

        result = ocr_with_openai(image_file, detail=detail, model=model)

        if result['success']:
            success_count += 1
            usage = result['usage']
            total_tokens += usage['total_tokens']

            if result.get('truncated', False):
                truncated_count += 1

            results[image_file.name] = {
                'text': result['text'],
                'model': result['model'],
                'detail': result['detail'],
                'usage': usage,
                'finish_reason': result['finish_reason'],
                'truncated': result.get('truncated', False)
            }

            print(f" ✅ Thành công!")
            print(f" - Tokens: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
            print(f" - Finish reason: {result['finish_reason']}")

            text_preview = result['text'][:150].replace('\n', ' ')
            print(f" - Preview: {text_preview}...")

        else:
            print(f" ❌ Lỗi: {result['error']}")
            results[image_file.name] = {
                'error': result['error'],
                'text': ''
            }

        print()
        if idx < len(image_files):
            time.sleep(delay)

    print("\n" + "="*80)
    print("💰 CHI PHÍ ƯỚC TÍNH")
    print("="*80)

    if model == "gpt-4o":
        input_price = 2.50 / 1_000_000
        output_price = 10.00 / 1_000_000
    elif model == "gpt-4o-mini":
        input_price = 0.15 / 1_000_000
        output_price = 0.60 / 1_000_000
    else:
        input_price = output_price = 0

    total_input_tokens = sum(r['usage']['prompt_tokens'] for r in results.values() if 'usage' in r)
    total_output_tokens = sum(r['usage']['completion_tokens'] for r in results.values() if 'usage' in r)

    input_cost = total_input_tokens * input_price
    output_cost = total_output_tokens * output_price
    total_cost = input_cost + output_cost

    print(f"Model: {model}")
    print(f"Detail: {detail}")
    print(f"\nTokens:")
    print(f" - Input tokens: {total_input_tokens:,}")
    print(f" - Output tokens: {total_output_tokens:,}")
    print(f" - Total tokens: {total_tokens:,}")

    print(f"\nChi phí:")
    print(f" - Input cost: ${input_cost:.4f}")
    print(f" - Output cost: ${output_cost:.4f}")
    print(f" - TOTAL COST: ${total_cost:.4f} (≈ {total_cost * 25000:,.0f} VNĐ)")
    print(f" - Cost/image: ${total_cost/len(image_files):.4f}")

    print(f"\nThống kê:")
    print(f" - Tổng ảnh: {len(image_files)}")
    print(f" - Thành công: {success_count}")
    print(f" - Lỗi: {len(image_files) - success_count}")

    if truncated_count > 0:
        print(f"\n⚠️ CẢNH BÁO:")
        print(f" - Số ảnh bị cắt output: {truncated_count}/{len(image_files)}")
        print(f" - Khuyến nghị: Chuyển sang gpt-4o hoặc tăng max_tokens")

    output_json = folder / f'{output_name}.json'

    save_data = {
        'metadata': {
            'model': model,
            'detail': detail,
            'total_images': len(image_files),
            'success_count': success_count,
            'truncated_count': truncated_count,
            'total_cost_usd': total_cost,
            'total_cost_vnd': total_cost * 25000,
            'total_tokens': total_tokens,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens
        },
        'results': results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    output_txt = folder / f'{output_name}.txt'

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"OPENAI VISION OCR RESULTS\n")
        f.write(f"Model: {model} | Detail: {detail}\n")
        f.write(f"Total Cost: ${total_cost:.4f} (≈ {total_cost * 25000:,.0f} VNĐ)\n")

        if truncated_count > 0:
            f.write(f"⚠️ WARNING: {truncated_count} images had truncated output\n")

        f.write(f"="*80 + "\n\n")

        for filename, data in results.items():
            f.write(f"{'='*80}\n")
            f.write(f"📄 {filename}\n")

            if 'truncated' in data and data['truncated']:
                f.write(f"⚠️ OUTPUT BỊ CẮT - VĂN BẢN CÓ THỂ THIẾU!\n")

            f.write(f"{'='*80}\n")

            if 'error' not in data:
                f.write(f"Tokens: {data['usage']['total_tokens']} | Finish: {data['finish_reason']}\n\n")
                f.write(data['text'])
            else:
                f.write(f"❌ LỖI: {data['error']}")

            f.write(f"\n\n")

    print(f"\n📁 Kết quả đã lưu:")
    print(f" - JSON: {output_json}")
    print(f" - TXT: {output_txt}")
    print("="*80 + "\n")
    
    save_data["json_file"] = str(output_json)
    save_data["txt_file"] = str(output_txt)

    return save_data

if __name__ == "__main__":
    FOLDER_PATH = "data/input/img/hoa_test"
    results = process_exam_with_openai(
        folder_path=FOLDER_PATH,
        detail="high",
        model="gpt-4o-mini",
        delay=0.5,
        output_name='openai_ocr_low'
    )
