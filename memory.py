import gc
import os
import psutil

def get_memory_usage(log):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / 1024 / 1024  # Convert bytes to MB
    return mem_usage_mb

# Having a lot of issues with my Raspberry Pi OOMing, so trying aggressively GCing
def garbage_collect(log):
    mem_before = get_memory_usage(log)
    gc.collect()
    mem_after = get_memory_usage(log)
    log(f"Memory Usage: {mem_before:.2f} MB GC to {mem_after:.2f} MB")