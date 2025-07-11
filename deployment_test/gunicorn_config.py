"""
Gunicorn configuration for production deployment on Container-Optimized OS.
Optimized for single VM with moderate traffic loads.
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8080"
backlog = 2048

# Worker processes - conservative for VM deployment
workers = min(3, (multiprocessing.cpu_count() * 2) + 1)

# Worker class and connections
worker_class = "gthread"
threads = 2
worker_connections = 1000

# Request handling
max_requests = 1000
max_requests_jitter = 50

# Timeouts
timeout = 30
keepalive = 2
graceful_timeout = 30

# Process naming
proc_name = "mlops-churn-api"

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process management
daemon = False
pidfile = None
umask = 0
user = None
group = None

# Performance optimizations
preload_app = True
sendfile = True
reuse_port = True

# Worker temporary directory (use RAM for better performance)
worker_tmp_dir = "/dev/shm"

# Environment-specific settings
if os.getenv("FLASK_ENV") == "development":
    workers = 1
    threads = 1
    reload = True
    loglevel = "debug"

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("🚀 MLOps Churn API is ready to serve requests")

def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"Worker {worker.pid} exited")

def on_exit(server):
    """Called just before the master process exits."""
    server.log.info("🛑 MLOps Churn API is shutting down")