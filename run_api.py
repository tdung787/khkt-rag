"""
Script to run the Quiz Management API

Usage:
    python run_api.py
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 STARTING QUIZ MANAGEMENT API")
    print("=" * 70)
    print()
    print("⚙️  System Configuration:")
    print("   • Quiz format: FIXED 10 questions, 15 minutes")
    print("   • Cannot be changed via API or parameters")
    print()
    print("📍 API will be available at:")
    print("   • Main: http://localhost:8000")
    print("   • Docs: http://localhost:8000/docs")
    print("   • Health: http://localhost:8000/health")
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
    print()
    print("📊 Stats Endpoints:")
    print("   • GET  /api/stats - Thống kê tổng quan")
    print()
    print("⌨️  Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8110,
        reload=True,  # Auto-reload khi code thay đổi
        log_level="info"
    )