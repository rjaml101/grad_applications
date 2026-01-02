import os

def print_file_size_info(filepath):
    file_size_bytes = os.path.getsize(filepath)
    print(f"The size of '{filepath}' is: {file_size_bytes} bytes")
    print(f"The size in kilobytes is: {file_size_bytes / 1024:.2f} KB")
    print(f"The size in megabytes is: {file_size_bytes / (1024 * 1024):.2f} MB")
    return file_size_bytes
