"""gunicorn_config.py - Tối ưu cho 2.25GB available"""
import multiprocessing

# Với 2.25GB available → có thể dùng 10-12 workers
workers = 12

# Timeout cho LLM calls
timeout = 120

# Bind address
bind = "0.0.0.0:8110"

# ⚡ QUAN TRỌNG: Đổi worker class
worker_class = 'uvicorn.workers.UvicornWorker'  # ← Thay đổi này!

# Logs
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Auto-restart workers
max_requests = 1000
max_requests_jitter = 50

# Preload app
preload_app = True

print(f"🚀 Starting with {workers} workers (optimized for 2.25GB available)")