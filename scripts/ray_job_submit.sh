set -x

ray job submit --no-wait --runtime-env=tasks/runtime_env/runtime_env.yaml -- "$@"