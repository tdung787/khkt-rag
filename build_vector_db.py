import os
import json
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ================== CONFIG ==================
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large default dimension
BATCH_SIZE = 100
COLLECTION_NAME = "KHTN_QA"

# Paths
INPUT_JSON = "data/input/json/nhiet_hoc_VL-lop10-E.json"
DATABASE_FOLDER = "database"
QDRANT_PATH = f"{DATABASE_FOLDER}/qdrant_storage"
CHECKPOINT_FILE = f"{DATABASE_FOLDER}/embedding_checkpoint.json"

# Pricing (per 1M tokens)
EMBEDDING_COST_PER_1M = 0.13

# ================== LOGGING SETUP ==================
def setup_logging():
    """Setup logging"""
    log_folder = Path(DATABASE_FOLDER) / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_folder / f"embedding_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ================== CHECKPOINT MANAGEMENT ==================
def load_checkpoint() -> set:
    """Load danh sách các ID đã embed"""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("embedded_ids", []))
    return set()

def save_checkpoint(embedded_ids: set):
    """Save checkpoint"""
    Path(DATABASE_FOLDER).mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "embedded_ids": list(embedded_ids),
            "last_updated": datetime.now().isoformat(),
            "total_embedded": len(embedded_ids)
        }, f, ensure_ascii=False, indent=2)

# ================== EMBEDDING WITH RETRY ==================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def get_embeddings_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Embed một batch texts với retry logic"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

# ================== MAIN PROCESS ==================
def main():
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU QUÁ TRÌNH EMBEDDING VÀ UPLOAD LÊN QDRANT")
    logger.info("=" * 70)
    
    # Check input JSON exists
    if not Path(INPUT_JSON).exists():
        logger.error(f"❌ Không tìm thấy file: {INPUT_JSON}")
        logger.error("   Hãy chạy script parse_questions_to_json.py trước!")
        return
    
    # Load questions from JSON
    logger.info(f"📂 Đọc câu hỏi từ: {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    
    logger.info(f"📊 Tổng số câu hỏi trong JSON: {len(all_questions)}")
    
    # Check for duplicate IDs - STOP if found
    logger.info("🔍 Kiểm tra duplicate IDs...")
    ids = [q["id"] for q in all_questions]
    duplicate_ids = [id for id in ids if ids.count(id) > 1]
    
    if duplicate_ids:
        logger.error("=" * 70)
        logger.error("❌ PHÁT HIỆN DUPLICATE IDs - DỪNG QUÁ TRÌNH")
        logger.error("=" * 70)
        logger.error(f"Tìm thấy {len(set(duplicate_ids))} ID trùng lặp:")
        
        for dup_id in sorted(set(duplicate_ids)):
            dup_questions = [q for q in all_questions if q["id"] == dup_id]
            logger.error(f"\n🔴 ID: {dup_id} - Xuất hiện {len(dup_questions)} lần:")
            for i, q in enumerate(dup_questions, 1):
                logger.error(f"   [{i}] Question: {q['question'][:60]}...")
        
        logger.error("\n💡 Giải pháp:")
        logger.error("   1. Kiểm tra lại dữ liệu JSON")
        logger.error("   2. Đảm bảo mỗi ID là duy nhất")
        raise ValueError(f"Found {len(set(duplicate_ids))} duplicate IDs. Fix them before embedding.")
    
    logger.info("✓ Không có duplicate IDs")
    
    # Initialize clients
    logger.info("🔌 Khởi tạo OpenAI và Qdrant clients...")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize Qdrant (local mode)
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    logger.info(f"✓ Qdrant client đã kết nối tới: {QDRANT_PATH}")
    
    # Create collection if not exists
    collections = qdrant_client.get_collections().collections
    collection_exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not collection_exists:
        logger.info(f"📦 Tạo collection mới: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE
            )
        )
        logger.info(f"✓ Đã tạo collection {COLLECTION_NAME}")
    else:
        logger.info(f"✓ Collection {COLLECTION_NAME} đã tồn tại")
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"   Số vectors hiện có: {collection_info.points_count}")
    
    # Load checkpoint
    embedded_ids = load_checkpoint()
    logger.info(f"📋 Đã embed trước đó: {len(embedded_ids)} câu hỏi")
    
    # Filter out already embedded questions
    questions_to_embed = [q for q in all_questions if q["id"] not in embedded_ids]
    logger.info(f"🆕 Cần embed: {len(questions_to_embed)} câu hỏi mới")
    
    if len(questions_to_embed) == 0:
        logger.info("✅ Tất cả câu hỏi đã được embed rồi!")
        return
    
    # Estimate cost (bao gồm cả explanation)
    total_chars = sum(
        len(q["question"]) + 
        len(q["correct_answer_text"]) + 
        len(q.get("explanation", ""))
        for q in questions_to_embed
    )
    total_tokens = total_chars * 0.75  # Rough estimate: 1 char ≈ 0.75 tokens for Vietnamese
    
    estimated_cost = (total_tokens / 1_000_000) * EMBEDDING_COST_PER_1M
    logger.info(f"💰 Ước tính:")
    logger.info(f"   - Tổng ký tự: {total_chars:,}")
    logger.info(f"   - Tổng tokens (ước): ~{total_tokens:,.0f}")
    logger.info(f"   - Chi phí: ~${estimated_cost:.4f} (¢{estimated_cost*100:.2f})")
    
    # Confirm before proceeding
    logger.info("\n⏸️  Nhấn Ctrl+C để dừng, hoặc chờ 3 giây để tiếp tục...")
    import time
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        logger.info("\n❌ Đã hủy bởi người dùng")
        return
    
    # Process in batches
    logger.info(f"\n🚀 Bắt đầu embedding với batch size = {BATCH_SIZE}")
    logger.info(f"📝 Embedding text format: question + correct_answer_text + explanation")
    
    total_batches = (len(questions_to_embed) + BATCH_SIZE - 1) // BATCH_SIZE
    points_to_upload = []
    
    for batch_idx in tqdm(range(0, len(questions_to_embed), BATCH_SIZE), 
                          desc="Embedding batches", 
                          total=total_batches):
        batch = questions_to_embed[batch_idx:batch_idx + BATCH_SIZE]
        
        # Prepare embedding texts (question + correct answer + explanation)
        embedding_texts = [
            f"{q['question']}\n" + 
            "\n".join([f"{k}. {v}" for k, v in q['options'].items()])
            for q in batch
        ]
        
        try:
            # Get embeddings
            embeddings = get_embeddings_batch(openai_client, embedding_texts)
            
            # Prepare points for Qdrant
            for question, embedding in zip(batch, embeddings):
                # Prepare payload - chỉ lấy các trường tồn tại
                payload = {
                    "id": question["id"],
                    "question": question["question"],
                    "options": question["options"],
                    "correct_answer": question["correct_answer"],
                    "correct_answer_text": question["correct_answer_text"],
                    "explanation": question.get("explanation", ""),
                    "subject": question.get("subject", "unknown")
                }
                
                point = PointStruct(
                    id=hash(question["id"]) & 0x7FFFFFFFFFFFFFFF,  # Convert to positive int
                    vector=embedding,
                    payload=payload
                )
                points_to_upload.append(point)
                embedded_ids.add(question["id"])
            
            # Upload batch to Qdrant
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upload[-len(batch):]
            )
            
            # Save checkpoint after each batch
            save_checkpoint(embedded_ids)
            
            logger.info(f"✓ Batch {batch_idx//BATCH_SIZE + 1}/{total_batches}: "
                       f"Đã embed và upload {len(batch)} câu hỏi")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi xử lý batch {batch_idx//BATCH_SIZE + 1}: {e}")
            logger.error("🛑 DỪNG TOÀN BỘ QUÁ TRÌNH DO LỖI EMBEDDING")
            raise
    
    # Final collection info
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    
    # Final statistics
    logger.info("=" * 70)
    logger.info("KẾT QUẢ")
    logger.info("=" * 70)
    logger.info(f"✓ Tổng số câu hỏi đã embed: {len(embedded_ids)}")
    logger.info(f"✓ Câu hỏi mới embed trong lần này: {len(questions_to_embed)}")
    logger.info(f"✓ Collection: {COLLECTION_NAME}")
    logger.info(f"✓ Số vectors trong Qdrant: {collection_info.points_count}")
    logger.info(f"✓ Qdrant storage: {QDRANT_PATH}")
    logger.info(f"✓ Checkpoint: {CHECKPOINT_FILE}")
    logger.info("=" * 70)
    logger.info("✅ HOÀN TẤT!")

# ================== CLI ==================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Người dùng dừng chương trình")
    except Exception as e:
        logger.error(f"\n❌ Lỗi nghiêm trọng: {e}", exc_info=True)