#!/bin/bash

# 用法
#
# * 查pool
#   scripts/query_tool.sh list-pools
#
# * 列出queries
#   scripts/query_tool.sh --pool standalone_rollout list
#   scripts/query_tool.sh --pool standalone_rollout list-finshed
#
#   输出
#   --------
#   queryid123456
#   --------
#
# * 查具体某个query详细信息
#   scripts/query_tool.sh --pool validation get queryid123456
#
#   输出
#   ---------
#   request_id: 0855145abbeb41e99a2033ccbac08422
#   query:
#     id: 0855145abbeb41e99a2033ccbac08422
#     idx: 0855145abbeb41e99a2033ccbac08422
#     input_ids: [9885, 400, 19, 18, 17, 16, 15159, 22, 19732, 16, 17, 18, 15159, 22,
#   ---------
#

export RAY_BACKEND_LOG_LEVEL=error
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
export SOCKET_PATH=/tmp/query_tool.sock

if [ ! -S "${SOCKET_PATH}" ]; then
  ray job submit --no-wait \
    --runtime-env="tasks/runtime_env/runtime_env.yaml" \
    -- \
    python3 "$SCRIPT_DIR/query_tool.py" daemon

  MAX_WAIT=30  # 最多等待时间（秒）
  WAITED=0
  while [ ! -S "${SOCKET_PATH}" ]; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "Error: daemon socket not available after ${MAX_WAIT} seconds."
        echo "please check job logs by 'ray logs <job_id>'"
        exit 1
    fi
    echo "Waiting for daemon to be ready... (${WAITED}s)"
    sleep 2
    WAITED=$((WAITED + 2))
  done
fi
echo "daemon is ready" 1>&2

python3 "$SCRIPT_DIR/query_tool.py" "$@"
