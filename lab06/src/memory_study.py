import psutil
import os

process = psutil.Process(
    os.getpid()
)

memory = process.memory_info().rss

print(
    "Memory Usage (bytes):",
    memory
)

print(
    "Memory Usage (MB):",
    memory / (1024 * 1024)
)