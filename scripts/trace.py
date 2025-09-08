#!/usr/bin/env python3
import subprocess
import re
import sys
from datetime import datetime

PROCESS_KEYWORD = "ray::TaskRunner.main"
TOTAL_DURATION = 3600  # 总采样时间（秒）
SEGMENT_DURATION = 600  # 每段采样时间（秒）
RATE = 10  # 采样频率（Hz）
MLX_UPLOAD_CMD = ["/opt/tiger/mlx_deploy/bin/mlx", "asset", "upload"]


def find_pid(keyword):
    try:
        ps_output = subprocess.check_output(["ps", "aux"], text=True)
        for line in ps_output.splitlines():
            if keyword in line and "grep" not in line:
                parts = re.split(r"\s+", line)
                pid = parts[1]
                return pid
    except subprocess.CalledProcessError as e:
        print("Error running ps:", e, file=sys.stderr)
    return None


def run_pyspy_segment(pid, segment_index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"profile_{segment_index}_{timestamp}.json"
    cmd = [
        "py-spy", "record", "-p", pid, "--rate",
        str(RATE), "--duration",
        str(SEGMENT_DURATION), "-o", output_file, "--nonblocking", "--idle", "--format", "speedscope"
    ]
    print(f"Running segment {segment_index}: {cmd}")
    subprocess.run(cmd)

    # 上传到 MLX
    upload_cmd = MLX_UPLOAD_CMD + [output_file]
    print(f"Uploading segment {segment_index} to MLX: {''.join(upload_cmd)}")
    subprocess.run(upload_cmd)


if __name__ == "__main__":
    pid = find_pid(PROCESS_KEYWORD)
    if not pid:
        print(f"No process found matching {PROCESS_KEYWORD}")
        sys.exit(1)

    print(f"Found PID: {pid}. Starting segmented profiling and upload...")

    num_segments = TOTAL_DURATION // SEGMENT_DURATION
    for i in range(1, num_segments + 1):
        run_pyspy_segment(pid, i)

    print("All segments completed and uploaded.")
