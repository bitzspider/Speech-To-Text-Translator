import os
import subprocess

print("Testing ffmpeg availability")
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

ffmpeg_path = os.path.join(current_dir, "ffmpeg.exe")
print(f"ffmpeg_path: {ffmpeg_path}")
print(f"Exists: {os.path.exists(ffmpeg_path)}")

if os.path.exists(ffmpeg_path):
    try:
        result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True)
        print(f"Exit code: {result.returncode}")
        print(f"Output: {result.stdout[:100]}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("ffmpeg.exe not found in current directory")

# Try using PATH
try:
    print("\nChecking if ffmpeg is in PATH:")
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    print(f"Exit code: {result.returncode}")
    print(f"Output: {result.stdout[:100]}")
except Exception as e:
    print(f"Error: {e}")

# List all files in current directory
print("\nFiles in current directory:")
for file in os.listdir(current_dir):
    print(f"- {file} ({os.path.getsize(os.path.join(current_dir, file))} bytes)") 