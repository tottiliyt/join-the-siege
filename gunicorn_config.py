import multiprocessing

# Bind to this socket
bind = "0.0.0.0:8000"

# Number of worker processes
# A good rule of thumb is 2-4 x number of CPU cores
workers = multiprocessing.cpu_count() * 2 + 1

# Worker type
worker_class = "sync"

# Timeout for worker processes
timeout = 120

# Maximum number of simultaneous clients
max_requests = 1000

# Maximum number of requests a worker will process before restarting
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload application code before forking
preload_app = True

# Process naming
proc_name = "document_classifier"
