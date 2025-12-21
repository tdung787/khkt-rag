"""
MASTER PIPELINE - XỬ LÝ TÀI LIỆU GIÁO DỤC TRẮC NGHIỆM
Chạy toàn bộ quy trình từ DOCX → Vector Database

Pipeline:
1. Extract images from DOCX
2. OCR images to text (OpenAI Vision API)
3. Parse text to questions JSON (Q)
4. Map answers to questions (Q→A)
5. Map explanations to questions (A→E)
6. Build vector database with embeddings

Author: tdung787
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import các module con
sys.path.insert(0, str(Path(__file__).parent))

# Import từ các file đã có
# NOTE: Đặt TẤT CẢ các file 1-6 vào cùng thư mục với master_pipeline.py
# File 3 và 5 cần dùng PHIÊN BẢN MỚI đã fix
from extract_images_from_docx import extract_images_in_order
from extract_text_from_images import process_exam_with_openai
from convert_txt_to_Q import parse_txt_to_json  # ← File 3 mới (fix regex)
from convert_Q_to_A import load_json, load_answer_key, map_answers, save_json
from convert_A_to_E import parse_explanations, map_explanations  # ← File 5 mới (fix format)
from build_vector_db import main as build_vector_db_main

# ==================== CONFIGURATION ====================
class PipelineConfig:
    """Cấu hình cho toàn bộ pipeline"""
    
    def __init__(self):
        # Input files
        self.docx_path = "data/input/docx/ester-lipid_hoa12.docx"
        self.answer_key_txt = "data/input/txt/ester-lipid_hoa12-A.txt"
        self.explanation_txt = "data/input/txt/ester-lipid_hoa12-E.txt"
        
        # Subject info
        self.subject = "Hóa học"
        self.subject_code = "ester-lipid_hoa12"
        
        # Output folders
        self.img_folder = "data/input/img/ester-lipid_hoa12"
        self.txt_folder = "data/input/txt"
        self.json_folder = "data/input/json"
        self.database_folder = "database"
        
        # OCR settings
        self.ocr_model = "gpt-4o-mini"  # hoặc "gpt-4o"
        self.ocr_detail = "high"  # "low" hoặc "high"
        self.ocr_delay = 0.5
        
        # Steps to run (có thể bỏ qua các bước đã chạy)
        self.run_step_1 = True  # Extract images
        self.run_step_2 = True  # OCR
        self.run_step_3 = True  # Parse to Q
        self.run_step_4 = True  # Map answers
        self.run_step_5 = True  # Map explanations
        self.run_step_6 = True  # Build vector DB
        
        # Auto-generated file names
        self.ocr_output_name = f"ocr_{self.subject_code}"
        self.base_filename = f"{self.subject_code}"

    def get_file_paths(self) -> Dict[str, str]:
        """Lấy tất cả đường dẫn file cần thiết"""
        return {
            "ocr_json": f"{self.img_folder}/{self.ocr_output_name}.json",
            "ocr_txt": f"{self.img_folder}/{self.ocr_output_name}.txt",
            "q_json": f"{self.json_folder}/{self.base_filename}_Q.json",
            "a_json": f"{self.json_folder}/{self.base_filename}_A.json",
            "e_json": f"{self.json_folder}/{self.base_filename}_E.json",
        }

# ==================== HELPER FUNCTIONS ====================
def print_section(title: str):
    """In header cho mỗi section"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_step(step_num: int, title: str):
    """In thông tin bước"""
    print(f"\n🔹 BƯỚC {step_num}: {title}")
    print("-" * 80)

def create_folders(config: PipelineConfig):
    """Tạo các thư mục cần thiết"""
    folders = [
        config.img_folder,
        config.txt_folder,
        config.json_folder,
        config.database_folder
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"✅ Đã tạo các thư mục cần thiết")

def log_pipeline_start(config: PipelineConfig):
    """Log thông tin bắt đầu pipeline"""
    print_section("🚀 MASTER PIPELINE - BẮT ĐẦU XỬ LÝ")
    print(f"⏰ Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📚 Môn học: {config.subject}")
    print(f"📄 Input DOCX: {config.docx_path}")
    print(f"🤖 OCR Model: {config.ocr_model} ({config.ocr_detail})")
    
    steps_to_run = []
    if config.run_step_1: steps_to_run.append("1-Extract")
    if config.run_step_2: steps_to_run.append("2-OCR")
    if config.run_step_3: steps_to_run.append("3-Parse")
    if config.run_step_4: steps_to_run.append("4-Answer")
    if config.run_step_5: steps_to_run.append("5-Explain")
    if config.run_step_6: steps_to_run.append("6-VectorDB")
    
    print(f"📋 Các bước sẽ chạy: {' → '.join(steps_to_run)}")

def log_pipeline_end(results: Dict[str, Any]):
    """Log thông tin kết thúc pipeline"""
    print_section("✅ PIPELINE HOÀN TẤT")
    print(f"⏰ Hoàn thành lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Tổng thời gian: {results['total_time']:.2f}s")
    
    if results.get('step_times'):
        print("\n📊 Thời gian từng bước:")
        for step, duration in results['step_times'].items():
            print(f"   {step}: {duration:.2f}s")
    
    print("\n📁 Output files:")
    for key, path in results.get('output_files', {}).items():
        if path and Path(path).exists():
            print(f"   ✓ {key}: {path}")

# ==================== PIPELINE STEPS ====================
def step_1_extract_images(config: PipelineConfig) -> bool:
    """Bước 1: Trích xuất ảnh từ DOCX"""
    print_step(1, "TRÍCH XUẤT ẢNH TỪ DOCX")
    
    try:
        if not Path(config.docx_path).exists():
            print(f"❌ Không tìm thấy file DOCX: {config.docx_path}")
            return False
        
        extract_images_in_order(config.docx_path, config.img_folder)
        
        # Kiểm tra kết quả
        image_files = list(Path(config.img_folder).glob("image_*.png")) + \
                      list(Path(config.img_folder).glob("image_*.jpg"))
        
        print(f"✅ Đã trích xuất {len(image_files)} ảnh vào {config.img_folder}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi bước 1: {e}")
        return False

def step_2_ocr_images(config: PipelineConfig) -> bool:
    """Bước 2: OCR ảnh thành text"""
    print_step(2, "OCR ẢNH THÀNH TEXT (OpenAI Vision API)")
    
    try:
        result = process_exam_with_openai(
            folder_path=config.img_folder,
            detail=config.ocr_detail,
            model=config.ocr_model,
            delay=config.ocr_delay,
            output_name=config.ocr_output_name
        )
        
        if result and result.get('metadata'):
            meta = result['metadata']
            print(f"✅ OCR hoàn tất:")
            print(f"   - Số ảnh: {meta['total_images']}")
            print(f"   - Thành công: {meta['success_count']}")
            print(f"   - Chi phí: ${meta['total_cost_usd']:.4f}")
            return True
        else:
            print("❌ OCR thất bại")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi bước 2: {e}")
        return False

def step_3_parse_to_questions(config: PipelineConfig) -> bool:
    """Bước 3: Parse text thành JSON câu hỏi"""
    print_step(3, "PARSE TEXT THÀNH JSON CÂU HỎI")
    
    try:
        paths = config.get_file_paths()
        input_txt = paths['ocr_txt']
        output_json = paths['q_json']
        
        if not Path(input_txt).exists():
            print(f"❌ Không tìm thấy file OCR text: {input_txt}")
            return False
        
        parse_txt_to_json(
            input_path=input_txt,
            output_path=output_json,
            subject=config.subject
        )
        
        # Kiểm tra kết quả
        with open(output_json, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        print(f"✅ Đã parse {len(questions)} câu hỏi → {output_json}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi bước 3: {e}")
        return False

def step_4_map_answers(config: PipelineConfig) -> bool:
    """Bước 4: Gán đáp án cho câu hỏi"""
    print_step(4, "GÁN ĐÁP ÁN CHO CÂU HỎI")
    
    try:
        paths = config.get_file_paths()
        input_json = paths['q_json']
        output_json = paths['a_json']
        
        if not Path(input_json).exists():
            print(f"❌ Không tìm thấy file JSON câu hỏi: {input_json}")
            return False
        
        if not Path(config.answer_key_txt).exists():
            print(f"❌ Không tìm thấy file đáp án: {config.answer_key_txt}")
            return False
        
        # Load data
        data = load_json(input_json)
        answer_key = load_answer_key(config.answer_key_txt)
        
        # Map answers
        mapped_data, missing = map_answers(data, answer_key)
        
        # Save
        save_json(mapped_data, output_json)
        
        print(f"✅ Đã gán đáp án cho {len(mapped_data)} câu")
        if missing:
            print(f"⚠️  Thiếu đáp án: {len(missing)} câu")
            print(f"   {missing[:5]}..." if len(missing) > 5 else f"   {missing}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi bước 4: {e}")
        return False

def step_5_map_explanations(config: PipelineConfig) -> bool:
    """Bước 5: Gán lời giải cho câu hỏi"""
    print_step(5, "GÁN LỜI GIẢI CHO CÂU HỎI")
    
    try:
        paths = config.get_file_paths()
        input_json = paths['a_json']
        output_json = paths['e_json']
        
        if not Path(input_json).exists():
            print(f"❌ Không tìm thấy file JSON với đáp án: {input_json}")
            return False
        
        if not Path(config.explanation_txt).exists():
            print(f"⚠️  Không tìm thấy file lời giải: {config.explanation_txt}")
            print(f"   Bỏ qua bước này, copy file A→E")
            # Copy A to E if no explanations
            data = load_json(input_json)
            save_json(data, output_json)
            return True
        
        # Load data
        data = load_json(input_json)
        explanation_map = parse_explanations(config.explanation_txt)
        
        # Map explanations
        new_data, mapped, missing = map_explanations(data, explanation_map)
        
        # Save
        save_json(new_data, output_json)
        
        print(f"✅ Đã gán lời giải: {len(mapped)}/{len(explanation_map)} câu")
        if missing:
            print(f"⚠️  Thiếu trong JSON: {len(missing)} câu")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi bước 5: {e}")
        return False

def step_6_build_vector_db(config: PipelineConfig) -> bool:
    """Bước 6: Build vector database"""
    print_step(6, "BUILD VECTOR DATABASE")
    
    try:
        paths = config.get_file_paths()
        input_json = paths['e_json']
        
        if not Path(input_json).exists():
            print(f"❌ Không tìm thấy file JSON hoàn chỉnh: {input_json}")
            return False
        
        # Temporarily modify the INPUT_JSON in build_vector_db module
        import build_vector_db
        original_input = build_vector_db.INPUT_JSON
        build_vector_db.INPUT_JSON = input_json
        
        # Run vector DB build
        build_vector_db_main()
        
        # Restore original
        build_vector_db.INPUT_JSON = original_input
        
        print(f"✅ Đã build vector database từ {input_json}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi bước 6: {e}")
        return False

# ==================== MAIN PIPELINE ====================
def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Chạy toàn bộ pipeline"""
    
    start_time = time.time()
    results = {
        'success': True,
        'step_times': {},
        'output_files': config.get_file_paths(),
        'errors': []
    }
    
    # Setup
    log_pipeline_start(config)
    create_folders(config)
    
    # Step 1: Extract images
    if config.run_step_1:
        step_start = time.time()
        success = step_1_extract_images(config)
        results['step_times']['Step 1 - Extract Images'] = time.time() - step_start
        if not success:
            results['success'] = False
            results['errors'].append("Step 1 failed")
            return results
    
    # Step 2: OCR
    if config.run_step_2:
        step_start = time.time()
        success = step_2_ocr_images(config)
        results['step_times']['Step 2 - OCR'] = time.time() - step_start
        if not success:
            results['success'] = False
            results['errors'].append("Step 2 failed")
            return results
    
    # Step 3: Parse to Q
    if config.run_step_3:
        step_start = time.time()
        success = step_3_parse_to_questions(config)
        results['step_times']['Step 3 - Parse Questions'] = time.time() - step_start
        if not success:
            results['success'] = False
            results['errors'].append("Step 3 failed")
            return results
    
    # Step 4: Map answers
    if config.run_step_4:
        step_start = time.time()
        success = step_4_map_answers(config)
        results['step_times']['Step 4 - Map Answers'] = time.time() - step_start
        if not success:
            results['success'] = False
            results['errors'].append("Step 4 failed")
            return results
    
    # Step 5: Map explanations
    if config.run_step_5:
        step_start = time.time()
        success = step_5_map_explanations(config)
        results['step_times']['Step 5 - Map Explanations'] = time.time() - step_start
        if not success:
            results['success'] = False
            results['errors'].append("Step 5 failed")
            return results
    
    # Step 6: Build vector DB
    if config.run_step_6:
        step_start = time.time()
        success = step_6_build_vector_db(config)
        results['step_times']['Step 6 - Vector DB'] = time.time() - step_start
        if not success:
            results['success'] = False
            results['errors'].append("Step 6 failed")
            return results
    
    # Calculate total time
    results['total_time'] = time.time() - start_time
    
    return results

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Tạo config
    config = PipelineConfig()
    
    # ========================================
    # TÙY CHỈNH CONFIG Ở ĐÂY
    # ========================================
    
    # Đường dẫn input
    config.docx_path = "data/input/docx/ester-lipid_hoa12-347-348.docx"
    config.answer_key_txt = "data/input/txt/ester-lipid_hoa12-A.txt"
    config.explanation_txt = "data/input/txt/ester-lipid_hoa12-E.txt"
    
    # Thông tin môn học
    config.subject = "Hóa học"
    config.subject_code = "ester-lipid_hoa12"
    
    # Output folders
    config.img_folder = "data/input/img/ester-lipid_hoa12"
    
    # OCR settings
    config.ocr_model = "gpt-4o-mini"  # "gpt-4o" cho chất lượng cao hơn
    config.ocr_detail = "high"  # "low" rẻ hơn, "high" chính xác hơn
    
    # Chọn các bước cần chạy (đặt False để bỏ qua)
    config.run_step_1 = True   # Extract images
    config.run_step_2 = True   # OCR
    config.run_step_3 = True   # Parse to Q
    config.run_step_4 = True   # Map answers
    config.run_step_5 = True   # Map explanations
    config.run_step_6 = True  # Build vector DB
    
    # ========================================
    # CHẠY PIPELINE
    # ========================================
    
    try:
        results = run_pipeline(config)
        
        if results['success']:
            log_pipeline_end(results)
        else:
            print("\n❌ PIPELINE THẤT BẠI")
            print(f"Lỗi: {results['errors']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline bị dừng bởi người dùng (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ LỖI NGHIÊM TRỌNG: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)