#!/usr/bin/env bash
# Run the predeclared WB-001 DGX R2 Latin-square matrix on the host.
set -euo pipefail
shopt -s nullglob

if (( $# < 4 )); then
  echo "usage: $0 EXPECTED_COMMIT EXPECTED_IMAGE_ID OUTPUT_ROOT CACHE_ROOT [RUN_ID ...]" >&2
  exit 2
fi

EXPECTED_COMMIT=$1
EXPECTED_IMAGE_ID=$2
OUTPUT_ROOT=$3
CACHE_ROOT=$4
shift 4
SELECTED_RUNS=("$@")
IMAGE=${WB001_IMAGE:-llm-scratch:env-001}
RUN_TIMEOUT_SECONDS=${WB001_RUN_TIMEOUT_SECONDS:-1800}
ROOT=$(git rev-parse --show-toplevel)
SCRIPT=$(realpath "$0")
VERIFIER="$ROOT/docs/experiments/evidence/verify_wb001_dgx.py"
WANDB_INSPECTOR="$ROOT/docs/experiments/evidence/inspect_wandb_offline.py"
MATRIX=(
  "1 1 disabled"
  "1 2 offline-off"
  "1 3 offline-on"
  "2 1 offline-on"
  "2 2 disabled"
  "2 3 offline-off"
  "3 1 offline-off"
  "3 2 offline-on"
  "3 3 disabled"
)

for requested in "${SELECTED_RUNS[@]}"; do
  known=false
  for row in "${MATRIX[@]}"; do
    read -r repetition position arm <<< "$row"
    [[ $requested != "r${repetition}-p${position}-${arm}" ]] || known=true
  done
  [[ $known == true ]] || { echo "unknown RUN_ID: $requested" >&2; exit 2; }
done

if [[ ! $RUN_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]]; then
  echo "WB001_RUN_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
[[ ! -e $OUTPUT_ROOT ]] || {
  echo "refusing to reuse output root: $OUTPUT_ROOT" >&2
  exit 4
}
[[ ! -e $CACHE_ROOT ]] || {
  echo "refusing to reuse cache root: $CACHE_ROOT" >&2
  exit 4
}
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"
OUTPUT_ROOT=$(realpath "$OUTPUT_ROOT")
CACHE_ROOT=$(realpath "$CACHE_ROOT")
case "$OUTPUT_ROOT/" in "$ROOT/"*) echo "output root must be outside the repository" >&2; exit 2 ;; esac
case "$CACHE_ROOT/" in "$ROOT/"*) echo "cache root must be outside the repository" >&2; exit 2 ;; esac

HEAD=$(git -C "$ROOT" rev-parse HEAD)
[[ $HEAD == "$EXPECTED_COMMIT" ]] || {
  echo "expected commit $EXPECTED_COMMIT, found $HEAD" >&2
  exit 3
}
[[ -z $(git -C "$ROOT" status --porcelain=v1 --untracked-files=all) ]] || {
  echo "WB-001 R2 requires a clean worktree" >&2
  exit 3
}
ACTUAL_IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[[ $ACTUAL_IMAGE_ID == "$EXPECTED_IMAGE_ID" ]] || {
  echo "expected image $EXPECTED_IMAGE_ID, found $ACTUAL_IMAGE_ID" >&2
  exit 3
}

docker image inspect "$IMAGE" > "$OUTPUT_ROOT/image-inspect.json"
docker run --rm --pull=never --gpus all --network none \
  --entrypoint python -v "$ROOT:/workspace:ro" -w /workspace "$IMAGE" \
  scripts/diagnose_environment.py --json --require-cuda --require-bf16 \
  > "$OUTPUT_ROOT/diagnose.json" 2> "$OUTPUT_ROOT/diagnose.stderr"
{
  printf 'schema_version=1\n'
  printf 'ticket=WB-001\n'
  printf 'measured_commit=%s\n' "$HEAD"
  printf 'image=%s\n' "$IMAGE"
  printf 'image_id=%s\n' "$ACTUAL_IMAGE_ID"
  printf 'source_root=%s\n' "$ROOT"
  printf 'output_root=%s\n' "$OUTPUT_ROOT"
  printf 'cache_root=%s\n' "$CACHE_ROOT"
  printf 'runner_sha256=%s\n' "$(sha256sum "$SCRIPT" | cut -d' ' -f1)"
  printf 'verifier_sha256=%s\n' "$(sha256sum "$VERIFIER" | cut -d' ' -f1)"
  printf 'wandb_inspector_sha256=%s\n' "$(sha256sum "$WANDB_INSPECTOR" | cut -d' ' -f1)"
} > "$OUTPUT_ROOT/matrix.env"

GPU_PID=
HOST_PID=
STATS_PID=
DOCKER_PID=
RUN_NAME=
cleanup() {
  local status=$?
  for pid in "$GPU_PID" "$HOST_PID" "$STATS_PID" "$DOCKER_PID"; do
    [[ -z $pid ]] || kill "$pid" 2>/dev/null || true
    [[ -z $pid ]] || wait "$pid" 2>/dev/null || true
  done
  [[ -z $RUN_NAME ]] || docker rm -f "$RUN_NAME" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

cache_digest() {
  find "$CACHE_ROOT" -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1
}

assert_idle_inputs() {
  local lease
  for lease in "$CACHE_ROOT"/.leases/*.lease; do
    flock -n "$lease" true || { echo "active cache lease: $lease" >&2; return 1; }
  done
  local processes
  processes=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits)
  [[ ! $processes =~ (^|$'\n')[[:space:]]*[0-9]+[[:space:]]*, ]] || {
    echo "competing GPU compute process detected: $processes" >&2
    return 1
  }
}

idle_temperature_check() {
  local out=$1
  : > "$out"
  local start_ns now_ns elapsed_ns cutoff_ns sample spread
  start_ns=$(date +%s%N)
  while true; do
    sample=$(nvidia-smi \
      --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,pstate \
      --format=csv,noheader,nounits)
    now_ns=$(date +%s%N)
    printf '%s,%s\n' "$now_ns" "$sample" >> "$out"
    elapsed_ns=$(( now_ns - start_ns ))
    if (( elapsed_ns >= 30000000000 )); then
      cutoff_ns=$(( now_ns - 30000000000 ))
      spread=$(awk -F',' -v cutoff="$cutoff_ns" '
        $1 >= cutoff {
          temperature=$4
          gsub(/[[:space:]]/, "", temperature)
          if (temperature ~ /^[0-9]+$/) {
            if (count == 0 || temperature < min) min=temperature
            if (count == 0 || temperature > max) max=temperature
            count++
          }
        }
        END { if (count >= 27) print max-min; else print "invalid:" count }
      ' "$out")
      if [[ $spread =~ ^[0-9]+$ ]] && (( spread <= 2 )); then
        return 0
      fi
    fi
    (( elapsed_ns >= 90000000000 )) && break
    sleep 1
  done
  echo "idle GPU trailing 30-second temperature evidence was '$spread' after 90 seconds" >&2
  return 1
}

selected() {
  local run_id=$1
  (( ${#SELECTED_RUNS[@]} == 0 )) && return 0
  local candidate
  for candidate in "${SELECTED_RUNS[@]}"; do [[ $candidate != "$run_id" ]] || return 0; done
  return 1
}

prime_cache() {
  local out="$OUTPUT_ROOT/cache-prime"
  [[ ! -e $out ]] || { echo "refusing to reuse cache-prime evidence: $out" >&2; return 4; }
  mkdir -p "$out"
  assert_idle_inputs
  local before after status
  before=$(cache_digest)
  RUN_NAME=llm-scratch-wb001-cache-prime
  set +e
  timeout --signal=TERM --kill-after=30 "$RUN_TIMEOUT_SECONDS" \
    docker run --rm --pull=never --gpus all --network none --ipc=host --name "$RUN_NAME" \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
      -e GIT_CONFIG_VALUE_0=/workspace -e PYTHONPATH=/workspace/src \
      -v "$ROOT:/workspace:ro" -v "$CACHE_ROOT:/cache" -v "$out:/evidence" \
      -w /workspace "$IMAGE" \
      python src/train.py profile=stability_smoke runtime.device=cuda \
        data.streaming.cache.dir=/cache reproducibility.seed=42 \
        data.streaming.train.max_tokens=133248 \
        training.sequence_length=64 model.num_layers=26 \
        training.max_steps=1 training.max_tokens=null training.max_time=null \
        artifacts.checkpoints_dir=/evidence/checkpoints measurement.enabled=false \
        wandb.mode=disabled wandb.watch.enabled=false wandb.artifact.policy=none \
        hydra.run.dir=/evidence/hydra \
      > "$out/stdout.log" 2> "$out/stderr.log"
  status=$?
  set -e
  docker rm -f "$RUN_NAME" >/dev/null 2>&1 || true
  RUN_NAME=
  after=$(cache_digest)
  {
    printf 'schema_version=1\nexit_code=%s\n' "$status"
    printf 'cache_before=%s\ncache_after=%s\n' "$before" "$after"
  } > "$out/conditions.env"
  find "$out" -type f ! -name artifact-sha256.txt -print0 | sort -z | \
    xargs -0 sha256sum > "$out/artifact-sha256.txt"
  (( status == 0 )) || { echo "cache prime failed with exit $status" >&2; return "$status"; }
  assert_idle_inputs
  printf 'cache_baseline_sha256=%s\n' "$after" >> "$OUTPUT_ROOT/matrix.env"
}

run_one() {
  local repetition=$1 position=$2 arm=$3
  local run_id="r${repetition}-p${position}-${arm}"
  selected "$run_id" || return 0
  local mode=offline watch=false
  [[ $arm != disabled ]] || mode=disabled
  [[ $arm != offline-on ]] || watch=true
  local out="$OUTPUT_ROOT/$run_id"
  [[ ! -e $out ]] || { echo "refusing to reuse evidence directory: $out" >&2; return 4; }
  mkdir -p "$out"/{wandb,wandb-cache,wandb-config,wandb-data}
  assert_idle_inputs
  local cache_before cache_after free_before free_after
  cache_before=$(cache_digest)
  free_before=$(df -B1 --output=avail "$out" | tail -n1 | tr -d ' ')
  idle_temperature_check "$out/idle-gpu.csv"

  RUN_NAME="llm-scratch-wb001-$run_id"
  local command=(
    docker run --rm --pull=never --gpus all --network none --ipc=host
    --ulimit memlock=-1 --ulimit stack=67108864 --name "$RUN_NAME"
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory
    -e GIT_CONFIG_VALUE_0=/workspace -e PYTHONPATH=/workspace/src
    -e WANDB_DIR=/evidence/wandb -e WANDB_CACHE_DIR=/evidence/wandb-cache
    -e WANDB_CONFIG_DIR=/evidence/wandb-config -e WANDB_DATA_DIR=/evidence/wandb-data
    -v "$ROOT:/workspace:ro" -v "$CACHE_ROOT:/cache" -v "$out:/evidence"
    -w /workspace "$IMAGE"
    sh -c 'while [ ! -f /evidence/START ]; do sleep 0.05; done; exec "$@"' sh
    python src/train.py profile=stability_smoke runtime.device=cuda
    data.streaming.cache.dir=/cache reproducibility.seed=42
    data.streaming.train.max_tokens=133248
    training.sequence_length=64 model.num_layers=26
    training.max_steps=260 training.max_tokens=null training.max_time=null
    artifacts.checkpoints_dir=/evidence/checkpoints
    measurement.enabled=true measurement.warmup_optimizer_steps=26
    measurement.cuda_events=true measurement.output_path=/evidence/measurement.json
    "wandb.mode=$mode" "wandb.watch.enabled=$watch" "wandb.name=$run_id"
    wandb.artifact.policy=none hydra.run.dir=/evidence/hydra
  )
  printf '%q ' "${command[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  {
    printf 'schema_version=1\nticket=WB-001\nrun_id=%s\n' "$run_id"
    printf 'repetition=%s\nposition=%s\narm=%s\nmode=%s\nwatch=%s\n' \
      "$repetition" "$position" "$arm" "$mode" "$watch"
    printf 'commit=%s\nimage_id=%s\ncache_before=%s\nfree_before_bytes=%s\n' \
      "$HEAD" "$ACTUAL_IMAGE_ID" "$cache_before" "$free_before"
  } > "$out/conditions.env"

  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu \
    --format=csv,noheader,nounits -lms 200 > "$out/gpu.csv" 2> "$out/gpu.stderr" &
  GPU_PID=$!
  vmstat -w -t 1 > "$out/host-vmstat.txt" 2> "$out/host-vmstat.stderr" &
  HOST_PID=$!
  timeout --signal=TERM --kill-after=30 "$RUN_TIMEOUT_SECONDS" "${command[@]}" \
    > "$out/stdout.log" 2> "$out/stderr.log" &
  DOCKER_PID=$!

  local attempt
  for attempt in $(seq 1 300); do
    docker inspect "$RUN_NAME" > "$out/container-inspect.json" 2>/dev/null && break
    kill -0 "$DOCKER_PID" 2>/dev/null || break
    sleep 0.1
  done
  [[ -s $out/container-inspect.json ]] || { echo "container was not inspectable" >&2; return 5; }
  (
    while docker inspect "$RUN_NAME" >/dev/null 2>&1; do
      sample=$(docker stats --no-stream \
        --format '{{.Container}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}|{{.NetIO}}|{{.BlockIO}}|{{.PIDs}}' \
        "$RUN_NAME") || break
      printf '%s|%s\n' "$(date +%s%N)" "$sample"
    done
  ) > "$out/container-stats.txt" 2> "$out/container-stats.stderr" &
  STATS_PID=$!
  for attempt in $(seq 1 150); do [[ -s $out/container-stats.txt ]] && break; sleep 0.1; done
  [[ -s $out/container-stats.txt ]] || { echo "container sampler did not start" >&2; return 5; }

  local start_ns end_ns status
  start_ns=$(date +%s%N)
  touch "$out/START"
  set +e
  wait "$DOCKER_PID"
  status=$?
  set -e
  DOCKER_PID=
  end_ns=$(date +%s%N)
  for pid in "$GPU_PID" "$HOST_PID" "$STATS_PID"; do kill "$pid" 2>/dev/null || true; done
  for pid in "$GPU_PID" "$HOST_PID" "$STATS_PID"; do wait "$pid" 2>/dev/null || true; done
  GPU_PID= HOST_PID= STATS_PID=
  docker rm -f "$RUN_NAME" >/dev/null 2>&1 || true
  RUN_NAME=
  docker run --rm --pull=never --network none --entrypoint python \
    -v "$ROOT:/workspace:ro" -v "$out:/evidence:ro" -w /workspace "$IMAGE" \
    docs/experiments/evidence/inspect_wandb_offline.py /evidence \
    > "$out/wandb-records.json"
  assert_idle_inputs
  cache_after=$(cache_digest)
  free_after=$(df -B1 --output=avail "$out" | tail -n1 | tr -d ' ')
  {
    printf 'start_unix_ns=%s\nend_unix_ns=%s\nexit_code=%s\n' "$start_ns" "$end_ns" "$status"
    printf 'cache_after=%s\ncache_unchanged=%s\nfree_after_bytes=%s\n' \
      "$cache_after" "$([[ $cache_before == "$cache_after" ]] && echo true || echo false)" "$free_after"
  } >> "$out/conditions.env"
  find "$out" -type f ! -name artifact-sha256.txt -print0 | sort -z | \
    xargs -0 sha256sum > "$out/artifact-sha256.txt"
  (( status == 0 )) || { echo "$run_id failed with exit $status; evidence retained" >&2; return "$status"; }
  local required
  for required in measurement.json hydra/resolved_config.yaml hydra/run_manifest.json \
    checkpoints/metrics.jsonl checkpoints/wandb_events.jsonl checkpoints/final.pt \
    wandb-records.json; do
    [[ -f $out/$required ]] || { echo "$run_id missing $required" >&2; return 6; }
  done
}

prime_cache
for row in "${MATRIX[@]}"; do read -r repetition position arm <<< "$row"; run_one "$repetition" "$position" "$arm"; done
trap - EXIT INT TERM
