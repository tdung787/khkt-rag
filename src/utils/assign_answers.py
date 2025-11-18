import os
import json
import logging
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from datetime import datetime

load_dotenv()

# ================== CONFIG ==================
MODEL = "gpt-4o-mini"
TEMPERATURE = 0  # Đảm bảo tính nhất quán
MIN_RESULT_LENGTH = 10  # Độ dài tối thiểu của kết quả hợp lệ
MAX_RETRIES = 3  # Số lần thử lại tối đa

SYSTEM_PROMPT = """\
Bạn là công cụ gán đáp án đúng vào các câu hỏi trắc nghiệm tiếng Việt.

Đầu vào:
- "Câu hỏi": chứa nhiều câu hỏi trắc nghiệm (Câu 1, Câu 2, …), mỗi câu có các lựa chọn A. B. C. D.
- "Đáp án": là phần văn bản riêng, có thể chứa một hoặc nhiều đáp án đúng, 
  không nhất thiết trùng số lượng hoặc thứ tự với câu hỏi.

Yêu cầu:
1. Đọc hiểu nội dung các câu hỏi và phần đáp án.
2. Với mỗi câu hỏi, xác định đáp án đúng (A, B, C, hoặc D) dựa theo ngữ nghĩa.
3. Chèn dòng `<Đáp án: X>` NGAY SAU dòng lựa chọn đúng đó.
4. Giữ nguyên HOÀN TOÀN định dạng gốc của câu hỏi, không thêm bất kỳ ký tự markdown hay formatting nào.
5. Nếu không tìm thấy đáp án phù hợp, bỏ trống (không chèn gì).

VÍ DỤ 1 - Đáp án là B:

Câu 1: Câu hỏi...
A. Lựa chọn A
B. Lựa chọn B (đây là đáp án đúng)
<Đáp án: B>
C. Lựa chọn C
D. Lựa chọn D

VÍ DỤ 2 - Đáp án là D:

Câu 2: Câu hỏi khác...
A. Lựa chọn A
B. Lựa chọn B
C. Lựa chọn C
D. Lựa chọn D (đây là đáp án đúng)
<Đáp án: D>

VÍ DỤ SAI - TUYỆT ĐỐI KHÔNG LÀM NHƯ VẬY:

Câu 3: ...
A. Lựa chọn A
<Đáp án: B>  ← SAI! Không được đặt trước dòng B
B. Lựa chọn B

LƯU Ý QUAN TRỌNG: 
- Tag <Đáp án: X> phải nằm NGAY SAU dòng lựa chọn đúng (cùng thứ tự với X)
- Nếu đáp án là C thì <Đáp án: C> phải nằm ngay sau dòng "C. ..."
- Nếu đáp án là D thì <Đáp án: D> phải nằm ngay sau dòng "D. ..."
- Tag <Đáp án: X> phải trên một dòng riêng, không nối liền với nội dung lựa chọn
- KHÔNG thêm bất kỳ dấu backtick (```), dấu ngoặc, hoặc ký tự markdown nào
- Chỉ output nội dung câu hỏi với tag <Đáp án: X> được chèn vào, không có gì khác
"""

# ================== LOGGING SETUP ==================
def setup_logging(output_folder):
    """Thiết lập logging với file và console"""
    log_folder = Path(output_folder) / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_folder / f"assignment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ================== API CALL WITH RETRY ==================
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_openai_api(client, q_text, a_text, model=MODEL):
    """Gọi OpenAI API với retry logic"""
    user_prompt = f"""\
Dưới đây là nội dung một trang bài tập trắc nghiệm và phần đáp án tương ứng.
Hãy đọc hiểu và chèn đáp án đúng vào vị trí phù hợp theo ngữ nghĩa.

--- CÂU HỎI ---
{q_text}

--- ĐÁP ÁN ---
{a_text}
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE
    )
    
    result = response.choices[0].message.content.strip()
    
    # Validate kết quả
    if not result or len(result) < MIN_RESULT_LENGTH:
        raise ValueError(f"Kết quả từ API quá ngắn hoặc rỗng: {len(result)} ký tự")
    
    return result

# ================== SAFE FILE OPERATIONS ==================
def safe_read_file(file_path):
    """Đọc file với xử lý lỗi encoding"""
    try:
        return Path(file_path).read_text(encoding="utf-8", errors='replace')
    except FileNotFoundError:
        raise
    except Exception as e:
        raise IOError(f"Lỗi đọc file {file_path}: {str(e)}")

def safe_write_file(file_path, content):
    """Ghi file với xử lý lỗi"""
    try:
        Path(file_path).write_text(content, encoding="utf-8", errors='replace')
        return True
    except Exception as e:
        raise IOError(f"Lỗi ghi file {file_path}: {str(e)}")

# ================== MAIN FUNCTION ==================
def assign_answers_with_ai(summary_json, questions_folder, answers_folder, output_folder, api_key=None):
    """
    Gán đáp án vào từng file câu hỏi dựa trên AI (GPT-4o-mini) với error handling nâng cao
    """
    # Setup logging
    logger = setup_logging(output_folder)
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU QUÁ TRÌNH GÁN ĐÁP ÁN BẰNG AI")
    logger.info("=" * 60)
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info(f"✓ Đã kết nối OpenAI API với model: {MODEL}")
    except Exception as e:
        logger.error(f"✗ Lỗi khởi tạo OpenAI client: {e}")
        return
    
    # Create output folder
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load summary
    try:
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
        logger.info(f"✓ Đã load file summary: {summary_json}")
    except Exception as e:
        logger.error(f"✗ Lỗi đọc file summary: {e}")
        return
    
    total = summary.get("total_files_processed", 0)
    
    # Trích xuất danh sách tất cả các file từ cấu trúc JSON
    all_files = []
    for th_key, th_data in summary.items():
        if th_key.startswith("TH"):  # TH1, TH2, TH3, TH4
            logger.info(f"📂 {th_key}: {th_data.get('count', 0)} files")
            
            # Lấy file từ "bat_dau_Cau"
            if "bat_dau_Cau" in th_data:
                files = th_data["bat_dau_Cau"].get("files", [])
                all_files.extend(files)
                logger.info(f"   └─ bat_dau_Cau: {len(files)} files")
            
            # Lấy file từ "khong_bat_dau_Cau"
            if "khong_bat_dau_Cau" in th_data:
                files = th_data["khong_bat_dau_Cau"].get("files", [])
                all_files.extend(files)
                logger.info(f"   └─ khong_bat_dau_Cau: {len(files)} files")
    
    # Remove duplicates và sort
    all_files = sorted(set(all_files))
    
    logger.info(f"📊 Tổng số file cần xử lý: {len(all_files)} (từ {total} file trong summary)")
    logger.info(f"   Sample files: {all_files[:3]}")
    
    if not all_files:
        logger.error("✗ CRITICAL: Không tìm thấy file nào để xử lý!")
        return
    
    # Statistics
    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "no_answer_file": 0
    }
    
    # Process each file with progress bar
    for fname in tqdm(all_files, desc="Xử lý file", unit="file"):
        q_file = Path(questions_folder) / fname
        a_file = Path(answers_folder) / fname
        out_file = out_path / fname
        
        # Check question file exists
        if not q_file.exists():
            logger.warning(f"⊘ SKIP - Không tìm thấy file câu hỏi: {fname}")
            stats["skipped"] += 1
            continue
        
        # Check answer file exists
        if not a_file.exists():
            logger.warning(f"⚠ WARNING - Không tìm thấy file đáp án: {fname}")
            a_text = ""
            stats["no_answer_file"] += 1
        else:
            try:
                a_text = safe_read_file(a_file)
            except Exception as e:
                logger.error(f"✗ ERROR - Lỗi đọc file đáp án {fname}: {e}")
                stats["failed"] += 1
                continue
        
        # Read question file
        try:
            q_text = safe_read_file(q_file)
        except Exception as e:
            logger.error(f"✗ ERROR - Lỗi đọc file câu hỏi {fname}: {e}")
            stats["failed"] += 1
            continue
        
        # Call AI API with retry
        try:
            result = call_openai_api(client, q_text, a_text)
            
            # Write result
            safe_write_file(out_file, result)
            logger.info(f"✓ SUCCESS - {fname} → Đã gán đáp án ({len(result)} ký tự)")
            stats["success"] += 1
            
        except ValueError as e:
            logger.error(f"✗ VALIDATION ERROR - {fname}: {e}")
            stats["failed"] += 1
            
        except Exception as e:
            logger.error(f"✗ ERROR - {fname}: {type(e).__name__} - {str(e)}")
            stats["failed"] += 1
    
    # Print final statistics
    logger.info("=" * 60)
    logger.info("KẾT QUẢ TỔNG KẾT")
    logger.info("=" * 60)
    logger.info(f"✓ Thành công: {stats['success']}/{len(all_files)}")
    logger.info(f"✗ Thất bại: {stats['failed']}/{len(all_files)}")
    logger.info(f"⊘ Bỏ qua: {stats['skipped']}/{len(all_files)}")
    logger.info(f"⚠ Không có file đáp án: {stats['no_answer_file']}/{len(all_files)}")
    logger.info(f"📁 Thư mục output: {output_folder}")
    logger.info("=" * 60)
    
    # Save statistics to JSON
    stats_file = out_path / "assignment_statistics.json"
    stats_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "total_files": len(all_files),
        "statistics": stats
    }
    
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Đã lưu thống kê vào: {stats_file}")
    except Exception as e:
        logger.error(f"✗ Lỗi lưu file thống kê: {e}")
    
    logger.info("\n✅ HOÀN TẤT QUÁ TRÌNH GÁN ĐÁP ÁN")


# ================== CLI ==================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Gán đáp án trắc nghiệm bằng AI với error handling nâng cao",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python script.py \\
    --summary_json data/output/page_comparison_summary.json \\
    --questions_folder data/input/cleaned_text \\
    --answers_folder data/input/normalized_answers \\
    --output_folder data/output/ai_assigned
        """
    )
    parser.add_argument("--summary_json", required=True, 
                       help="Đường dẫn tới file JSON thống kê")
    parser.add_argument("--questions_folder", required=True, 
                       help="Thư mục chứa file câu hỏi")
    parser.add_argument("--answers_folder", required=True, 
                       help="Thư mục chứa file đáp án")
    parser.add_argument("--output_folder", required=True, 
                       help="Thư mục lưu kết quả")
    parser.add_argument("--api_key", default=None, 
                       help="Tuỳ chọn: API key OpenAI (nếu không dùng biến môi trường)")
    
    args = parser.parse_args()

    assign_answers_with_ai(
        summary_json=args.summary_json,
        questions_folder=args.questions_folder,
        answers_folder=args.answers_folder,
        output_folder=args.output_folder,
        api_key=args.api_key
    )