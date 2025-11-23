"""gunicorn_config.py - Tối ưu cho 2.25GB available"""
import multiprocessing

# Với 2.25GB available → có thể dùng 10-12 workers
workers = 12  # Mỗi worker ~150-180MB

# Timeout cho LLM calls
timeout = 120

# Bind address
bind = "0.0.0.0:8110"

# Logs
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Auto-restart workers
max_requests = 1000
max_requests_jitter = 50

# Preload app (tiết kiệm RAM bằng cách share code)
preload_app = True

# Worker class
worker_class = 'sync'

print(f"🚀 Starting with {workers} workers (optimized for 2.25GB available)")