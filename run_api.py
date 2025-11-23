"""
Script to run the Quiz Management API with Gunicorn workers
Usage:
    python run_api.py           # Production mode (uses gunicorn_config.py)
    python run_api.py dev       # Development mode (auto-reload, 1 worker)
"""
import subprocess
import sys
import os

def print_banner():
    """Print startup banner"""
    print("=" * 70)
    print("🚀 STARTING QUIZ MANAGEMENT API")
    print("=" * 70)
    print()
    print("⚙️  System Configuration:")
    print("   • Quiz format: FIXED 10 questions, 15 minutes")
    print("   • Cannot be changed via API or parameters")
    print()
    print("📍 API will be available at:")
    print("   • Main: http://localhost:8110")
    print("   • Docs: http://localhost:8110/docs")
    print("   • Health: http://localhost:8110/health")
    print()
    print("📚 Quiz Endpoints:")
    print("   • GET  /api/quiz/latest - Bài kiểm tra mới nhất")
    print("   • GET  /api/quiz/all - Tất cả bài kiểm tra")
    print("   • GET  /api/quiz/{quiz_id} - Chi tiết 1 bài")
    print("   • GET  /api/quiz/current-status - Check quiz đang làm")
    print("   • GET  /api/quiz/daily-count - Thống kê theo ngày")
    print("   • GET  /api/quiz/by-date - Lấy bài theo ngày")
    print()
    print("📝 Submission Endpoints:")
    print("   • POST /api/submission/submit - Nộp bài và chấm điểm")
    print("   • GET  /api/submission/{id} - Thông tin bài nộp")
    print("   • GET  /api/submission/{id}/result - Kết quả chi tiết")
    print("   • GET  /api/submission/student/{id} - Lịch sử nộp bài")
    print("   • GET  /api/submission/all - Tất cả bài nộp")
    print()
    print("💬 Chat Session Endpoints:")
    print("   • POST /api/session/create - Tạo session mới")
    print("   • GET  /api/session/list - Danh sách sessions")
    print("   • GET  /api/session - Chi tiết session + history")
    print("   • DELETE /api/session - Xóa session")
    print()
    print("🤖 RAG Endpoints:")
    print("   • POST /api/rag/query - Hỏi chatbot (text + image)")
    print()
    print("📊 Stats Endpoints:")
    print("   • GET  /api/stats - Thống kê tổng quan")
    print("   • GET  /api/stats/daily - Đánh giá theo ngày")
    print("   • GET  /api/stats/history - Lịch sử đánh giá")
    print()

if __name__ == "__main__":
    print_banner()
    
    # Check mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    
    if mode == "dev":
        print("🔧 Running in DEVELOPMENT mode")
        print("   • Auto-reload: ON")
        print("   • Workers: 1 (for debugging)")
        print("   • Log level: DEBUG")
        print()
        print("⌨️  Press Ctrl+C to stop")
        print("=" * 70)
        print()
        
        # Development: single worker with reload
        subprocess.run([
            "gunicorn",
            "api.app:app",
            "--bind", "0.0.0.0:8110",
            "--workers", "1",
            "--reload",
            "--log-level", "debug",
            "--timeout", "120",
            "--access-logfile", "-",
            "--error-logfile", "-"
        ])
    else:
        print("🚀 Running in PRODUCTION mode")
        print("   • Config: gunicorn_config.py")
        print("   • Workers: 12 (defined in config)")
        print("   • RAM usage: ~2.0GB")
        print("   • Concurrent requests: 12")
        print()
        print("⌨️  Press Ctrl+C to stop")
        print("=" * 70)
        print()
        
        # Production: use gunicorn_config.py
        subprocess.run([
            "gunicorn",
            "src.api.app:app",
            "--config", "gunicorn_config.py"
        ])