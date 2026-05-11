#!/usr/bin/env bash
# Atenia RTX 3090 / Linux / WSL benchmark battery.
#
# Intended use: clone/copy the repo, point MODELS_ROOT at the models folder,
# run this script, and bring back the generated bench_logs directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS_ROOT="${MODELS_ROOT:-}"
if [[ -z "${WORK_ROOT:-}" ]]; then
    if [[ -d /mnt/d ]]; then
        WORK_ROOT="/mnt/d/Atenia"
    else
        WORK_ROOT="/home/${USER:-atenia}/Atenia"
    fi
fi
SUITE="${SUITE:-quick}"
BATTERY_MODE="${BATTERY_MODE:-smokes}"
MAX_TOKENS="${MAX_TOKENS:-32}"
PROMPT="${PROMPT:-Tell me about the history of Rome}"
SKIP_BUILD="${SKIP_BUILD:-false}"
LIST_ONLY="${LIST_ONLY:-false}"
AUTO_MODE="${AUTO_MODE:-false}"
CLEAN_MODE="${CLEAN_MODE:-false}"
CLEAN_DEPS="${CLEAN_DEPS:-false}"
CLEAN_CUDA="${CLEAN_CUDA:-false}"
YES="${YES:-false}"
AUTO_INSTALL="${AUTO_INSTALL:-true}"
INSTALL_CUDA="${INSTALL_CUDA:-auto}"
CLI_FORCE_BF16="${CLI_FORCE_BF16:-false}"
TRACE="${TRACE:-false}"
EXE_PATH="${EXE_PATH:-}"

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'
NC=$'\033[0m'

log_info() { printf '%s[INFO]%s %s\n' "$BLUE" "$NC" "$*" >&2; }
log_success() { printf '%s[SUCCESS]%s %s\n' "$GREEN" "$NC" "$*" >&2; }
log_warning() { printf '%s[WARNING]%s %s\n' "$YELLOW" "$NC" "$*" >&2; }
log_error() { printf '%s[ERROR]%s %s\n' "$RED" "$NC" "$*" >&2; }
trace() { [[ "$TRACE" == "true" ]] && printf '%s[TRACE]%s %s\n' "$BLUE" "$NC" "$*" >&2; }
die() { log_error "$*"; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  ./scripts/rtx3090_battery.sh [options]

Options:
  --models-root PATH     Model root directory
  --work-root PATH       Work/output directory
  --suite quick|full     Battery size
  --mode smokes|cli|both What to run
  --max-tokens N         Tokens per generation
  --prompt TEXT          Prompt for all runs
  --skip-build           Reuse existing executable
  --exe PATH             Executable path when skipping build
  --list-only            Print cases and exit
  --auto                 Non-interactive autodetect mode
  --clean                Remove generated work artifacts and exit
  --clean-deps           Remove helper-installed userspace dependencies and exit
  --clean-cuda           With --clean-deps, also purge Ubuntu CUDA toolkit packages
  --yes                  Confirm destructive cleanup prompts
  --no-auto-install      Do not install missing userspace tools
  --install-cuda         Install Ubuntu CUDA toolkit when missing
  --no-install-cuda      Do not install Ubuntu CUDA toolkit automatically
  --cli-force-bf16       Force ATENIA_M8_BF16_KERNEL=1 for CLI probes
  --trace                Print script trace diagnostics
  -h, --help             Show help

Environment variables with the same names are also supported:
  MODELS_ROOT, WORK_ROOT, SUITE, MAX_TOKENS, PROMPT, SKIP_BUILD,
  BATTERY_MODE, LIST_ONLY, AUTO_MODE, CLEAN_MODE, CLEAN_DEPS,
  CLEAN_CUDA, YES, AUTO_INSTALL, INSTALL_CUDA,
  CLI_FORCE_BF16, TRACE, EXE_PATH
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --models-root) MODELS_ROOT="$2"; shift 2 ;;
            --work-root) WORK_ROOT="$2"; shift 2 ;;
            --suite) SUITE="$2"; shift 2 ;;
            --mode) BATTERY_MODE="$2"; shift 2 ;;
            --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
            --prompt) PROMPT="$2"; shift 2 ;;
            --skip-build) SKIP_BUILD=true; shift ;;
            --exe) EXE_PATH="$2"; shift 2 ;;
            --list-only) LIST_ONLY=true; shift ;;
            --auto) AUTO_MODE=true; shift ;;
            --clean) CLEAN_MODE=true; shift ;;
            --clean-deps) CLEAN_DEPS=true; shift ;;
            --clean-cuda) CLEAN_CUDA=true; shift ;;
            --yes) YES=true; shift ;;
            --no-auto-install) AUTO_INSTALL=false; shift ;;
            --install-cuda) INSTALL_CUDA=true; shift ;;
            --no-install-cuda) INSTALL_CUDA=false; shift ;;
            --cli-force-bf16) CLI_FORCE_BF16=true; shift ;;
            --trace) TRACE=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown option: $1" ;;
        esac
    done
}

new_dir() {
    mkdir -p "$1"
}

source_cargo_env() {
    if ! command -v cargo >/dev/null 2>&1 && [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1090
        source "$HOME/.cargo/env"
    fi
}

have_sudo() {
    command -v sudo >/dev/null 2>&1
}

apt_install() {
    have_sudo || die "sudo is required to install missing packages: $*"
    log_info "Installing: $*"
    sudo apt-get update
    sudo apt-get install -y "$@"
}

install_rustup() {
    command -v curl >/dev/null 2>&1 || apt_install curl ca-certificates
    log_info "Installing Rust via rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source_cargo_env
}

cuda_toolkit_present() {
    command -v nvcc >/dev/null 2>&1 && {
        [[ -f /usr/local/cuda/include/cuda_runtime.h ]] || [[ -f /usr/include/cuda_runtime.h ]]
    }
}

should_install_cuda() {
    [[ "$INSTALL_CUDA" == "true" ]] && return 0
    [[ "$INSTALL_CUDA" == "false" ]] && return 1
    [[ "$AUTO_INSTALL" == "true" ]] || return 1
    return 0
}

install_cuda_toolkit() {
    should_install_cuda || return 0
    cuda_toolkit_present && return 0

    have_sudo || die "sudo is required to install CUDA toolkit packages"
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required before installing CUDA toolkit; install/update the host NVIDIA driver first"

    log_info "Installing Ubuntu CUDA toolkit packages"
    sudo apt-get update
    if ! sudo apt-get install -y nvidia-cuda-toolkit; then
        log_warning "nvidia-cuda-toolkit was not available from current apt sources"
        if [[ "$INSTALL_CUDA" == "true" ]]; then
            die "CUDA toolkit install failed. Add NVIDIA CUDA apt repository for this Ubuntu release, then rerun."
        fi
    fi
}

ensure_tools() {
    source_cargo_env

    if [[ "$AUTO_INSTALL" == "true" ]]; then
        command -v cc >/dev/null 2>&1 || apt_install build-essential pkg-config
        command -v jq >/dev/null 2>&1 || apt_install jq
        command -v cargo >/dev/null 2>&1 || install_rustup
        install_cuda_toolkit
    fi

    local missing=()
    command -v cargo >/dev/null 2>&1 || missing+=("cargo")
    command -v jq >/dev/null 2>&1 || missing+=("jq")
    command -v nvidia-smi >/dev/null 2>&1 || missing+=("nvidia-smi")
    cuda_toolkit_present || missing+=("cuda-toolkit")

    if ((${#missing[@]} > 0)); then
        die "Missing required tools: ${missing[*]}. nvidia-smi must be provided by the NVIDIA driver; cargo/jq/build tools/CUDA toolkit can be installed automatically with AUTO_INSTALL=true."
    fi
}

detect_mounted_disks() {
    lsblk -o MOUNTPOINT -n -l 2>/dev/null | awk 'NF' | sort -u || true
}

find_model_roots() {
    local candidates=()

    [[ -n "$MODELS_ROOT" ]] && candidates+=("$MODELS_ROOT")
    candidates+=(
        "/mnt/d/models"
        "/mnt/f/models"
        "/mnt/models"
        "$HOME/models"
        "$REPO_ROOT/models"
    )

    local c
    for c in "${candidates[@]}"; do
        [[ -d "$c" ]] && printf '%s\n' "$c"
    done

    find /mnt "$HOME" -maxdepth 3 -type d -name models 2>/dev/null || true
}

resolve_models_root() {
    if [[ -n "$MODELS_ROOT" && -d "$MODELS_ROOT" ]]; then
        return
    fi

    local roots
    mapfile -t roots < <(find_model_roots | awk '!seen[$0]++')

    if ((${#roots[@]} == 1)); then
        MODELS_ROOT="${roots[0]}"
        log_info "Auto-detected MODELS_ROOT=$MODELS_ROOT"
        return
    fi

    if [[ "$AUTO_MODE" == "true" || ! -t 0 ]]; then
        if ((${#roots[@]} > 0)); then
            MODELS_ROOT="${roots[0]}"
            log_info "Auto-selected MODELS_ROOT=$MODELS_ROOT"
            return
        fi
        die "MODELS_ROOT is not set and no models directory was found"
    fi

    echo
    log_info "Detected mount points:"
    detect_mounted_disks | sed 's/^/  /'
    echo

    if ((${#roots[@]} > 0)); then
        log_info "Potential models directories:"
        local i=1
        for r in "${roots[@]}"; do
            printf '  [%d] %s\n' "$i" "$r"
            i=$((i + 1))
        done
        printf '  [0] Enter manually\n'
        read -r -p "Select models directory (0-${#roots[@]}): " choice
        if [[ "$choice" =~ ^[0-9]+$ && "$choice" -ge 1 && "$choice" -le ${#roots[@]} ]]; then
            MODELS_ROOT="${roots[$((choice - 1))]}"
            return
        fi
    fi

    read -r -p "MODELS_ROOT path: " MODELS_ROOT
}

prompt_with_default() {
    local label="$1" default="$2" value
    read -r -p "$label [$default]: " value
    printf '%s' "${value:-$default}"
}

configure_interactive_run() {
    [[ "$AUTO_MODE" == "true" || ! -t 0 ]] && return
    [[ "$LIST_ONLY" == "true" || "$CLEAN_MODE" == "true" ]] && return

    echo
    log_info "Run configuration"
    printf '  [1] repo smokes\n'
    printf '  [2] CLI generation battery\n'
    printf '  [3] both\n'
    read -r -p "Select run mode (1-3) [1]: " mode_choice
    case "${mode_choice:-1}" in
        1|smokes|smoke) BATTERY_MODE="smokes" ;;
        2|cli|generation) BATTERY_MODE="cli" ;;
        3|both) BATTERY_MODE="both" ;;
        *) log_warning "Unknown run mode '$mode_choice'; using smokes"; BATTERY_MODE="smokes" ;;
    esac

    printf '  [1] normal / quick\n'
    printf '  [2] full\n'
    read -r -p "Select suite (1-2) [1]: " suite_choice
    case "${suite_choice:-1}" in
        1|quick|normal) SUITE="quick" ;;
        2|full) SUITE="full" ;;
        *) log_warning "Unknown suite choice '$suite_choice'; using quick"; SUITE="quick" ;;
    esac

    if [[ "$BATTERY_MODE" == "cli" || "$BATTERY_MODE" == "both" ]]; then
        MAX_TOKENS="$(prompt_with_default "Output tokens" "$MAX_TOKENS")"
        if ! [[ "$MAX_TOKENS" =~ ^[0-9]+$ && "$MAX_TOKENS" -gt 0 ]]; then
            log_warning "Invalid token count '$MAX_TOKENS'; using 32"
            MAX_TOKENS=32
        fi
        PROMPT="$(prompt_with_default "Prompt" "$PROMPT")"
    fi
    WORK_ROOT="$(prompt_with_default "Work/output directory" "$WORK_ROOT")"

    read -r -p "Skip build and reuse existing binary? [y/N]: " skip_choice
    case "${skip_choice:-n}" in
        y|Y|yes|YES|s|S|si|SI) SKIP_BUILD=true ;;
        *) SKIP_BUILD=false ;;
    esac
}

clean_generated_artifacts() {
    log_info "Cleaning generated artifacts under $WORK_ROOT"
    rm -rf \
        "$WORK_ROOT/cargo-target-rtx3090" \
        "$WORK_ROOT/runtime-cache" \
        "$WORK_ROOT/bench_logs"
    log_success "Clean complete"
}

confirm_cleanup() {
    local message="$1"
    if [[ "$YES" == "true" ]]; then
        return 0
    fi
    if [[ ! -t 0 ]]; then
        die "$message Re-run with --yes to confirm in non-interactive mode."
    fi
    echo
    log_warning "$message"
    read -r -p "Type DELETE to continue: " answer
    [[ "$answer" == "DELETE" ]] || die "Cleanup cancelled"
}

apt_purge_if_present() {
    local packages=("$@")
    have_sudo || die "sudo is required to purge packages: ${packages[*]}"
    sudo apt-get remove --purge -y "${packages[@]}" || true
}

clean_dependency_artifacts() {
    confirm_cleanup "This will remove rustup cargo (~/.cargo, ~/.rustup), purge jq/pkg-config/build-essential, and clean apt caches."

    log_info "Removing rustup/cargo directories"
    rm -rf "$HOME/.cargo" "$HOME/.rustup"

    log_info "Purging helper userspace apt packages"
    apt_purge_if_present jq pkg-config build-essential

    if [[ "$CLEAN_CUDA" == "true" ]]; then
        confirm_cleanup "This will also purge Ubuntu CUDA toolkit packages. It will not remove the Windows/host NVIDIA driver used by WSL nvidia-smi."
        log_info "Purging Ubuntu CUDA toolkit packages"
        apt_purge_if_present 'cuda*' 'libcudnn*' 'nsight*' nvidia-cuda-toolkit cuda-keyring
        sudo rm -f /etc/apt/sources.list.d/cuda*.list || true
    fi

    log_info "Running apt autoremove/clean"
    sudo apt-get autoremove -y || true
    sudo apt-get clean || true
    hash -r || true
    log_success "Dependency cleanup complete"
}

json_or_empty_object() {
    local s="$1"
    if printf '%s' "$s" | jq -e . >/dev/null 2>&1; then
        printf '%s' "$s"
    else
        printf '{}'
    fi
}

extract_error_summary() {
    local stderr_path="$1" line
    line="$(
        grep -E '^(error:|thread .+ panicked|panicked at|fatal:|CUDA error|cu[A-Za-z].*failed|.*failed to load model:)' "$stderr_path" 2>/dev/null \
          | head -n1 \
          | sed 's/[[:space:]]\+/ /g'
    )"
    if [[ -z "$line" ]]; then
        line="$(tail -n 1 "$stderr_path" 2>/dev/null | sed 's/[[:space:]]\+/ /g')"
    fi
    printf '%s' "$line"
}

tail_nonempty_line() {
    local path="$1"
    awk 'NF {line=$0} END {gsub(/[[:space:]]+/, " ", line); print line}' "$path" 2>/dev/null | cut -c1-180
}

run_with_heartbeat() {
    local label="$1" stdout_path="$2" stderr_path="$3"
    shift 3

    set +e
    "$@" >"$stdout_path" 2>"$stderr_path" &
    local pid=$!
    local start elapsed last_stdout last_stderr
    start="$(date +%s)"

    while kill -0 "$pid" 2>/dev/null; do
        sleep 10
        elapsed=$(( $(date +%s) - start ))
        last_stdout="$(tail_nonempty_line "$stdout_path")"
        last_stderr="$(tail_nonempty_line "$stderr_path")"
        if [[ -n "$last_stderr" ]]; then
            log_info "[$label] running ${elapsed}s | stderr: $last_stderr"
        elif [[ -n "$last_stdout" ]]; then
            log_info "[$label] running ${elapsed}s | stdout: $last_stdout"
        else
            log_info "[$label] running ${elapsed}s | waiting for first output..."
        fi
    done

    wait "$pid"
    local code=$?
    set -u
    return "$code"
}

get_nvidia_snapshot() {
    local smi query gpu_json
    smi="$(nvidia-smi 2>&1 || true)"
    query="$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.limit --format=csv,noheader,nounits 2>/dev/null || true)"
    gpu_json="$(
        printf '%s\n' "$query" | jq -Rn '
          [inputs
           | select(length > 0)
           | split(",")
           | map(gsub("^\\s+|\\s+$"; ""))
           | {
               name: (.[0] // ""),
               driver_version: (.[1] // ""),
               memory_total_mib: ((.[2] // "0") | tonumber? // 0),
               memory_free_mib: ((.[3] // "0") | tonumber? // 0),
               temperature_c: ((.[4] // "0") | tonumber? // 0),
               power_limit_w: ((.[5] // "0") | tonumber? // 0)
             }]
        ' 2>/dev/null
    )"
    jq -n --arg smi "$smi" --arg gpu_json "${gpu_json:-[]}" \
      '{nvidia_smi:$smi, gpu_query:($gpu_json | fromjson? // []), driver_query_error:null}'
}

get_system_snapshot() {
    local ram_total_kib ram_free_kib cpu os
    ram_total_kib="$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    ram_free_kib="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    cpu="$(lscpu 2>/dev/null | awk -F: '/Model name/{sub(/^[ \t]+/, "", $2); print $2; exit}')"
    os="$(lsb_release -d 2>/dev/null | cut -f2- || uname -a)"

    jq -n \
      --arg host "$(hostname)" \
      --arg ts "$(date -Iseconds)" \
      --arg os "$os" \
      --arg kernel "$(uname -r)" \
      --arg cpu "${cpu:-unknown}" \
      --arg lp "$(nproc 2>/dev/null || echo 0)" \
      --arg rt "$ram_total_kib" \
      --arg rf "$ram_free_kib" \
      --arg models "$MODELS_ROOT" \
      --arg work "$WORK_ROOT" \
      --arg suite "$SUITE" \
      --arg mt "$MAX_TOKENS" \
      --arg prompt "$PROMPT" \
      '{
        computer_name:$host,
        timestamp:$ts,
        os_caption:$os,
        os_version:$kernel,
        cpu_name:$cpu,
        logical_processors:($lp | tonumber? // 0),
        ram_total_gib:(($rt | tonumber? // 0) / 1048576),
        ram_free_gib:(($rf | tonumber? // 0) / 1048576),
        models_root:$models,
        work_root:$work,
        suite:$suite,
        max_tokens:($mt | tonumber? // 0),
        prompt:$prompt
      }'
}

get_repo_snapshot() {
    local branch commit dirty status
    branch="$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    commit="$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || true)"
    status="$(cd "$REPO_ROOT" && git status --short 2>/dev/null || true)"
    dirty=false
    [[ -n "$status" ]] && dirty=true

    jq -n \
      --arg branch "$branch" \
      --arg commit "$commit" \
      --arg dirty "$dirty" \
      --arg status "$status" \
      '{branch:$branch, commit:$commit, dirty:($dirty == "true"), status_short:$status}'
}

get_tools_snapshot() {
    jq -n \
      --arg cargo "$(cargo --version 2>/dev/null || true)" \
      --arg rustc "$(rustc --version 2>/dev/null || true)" \
      --arg jqv "$(jq --version 2>/dev/null || true)" \
      --arg gcc "$(cc --version 2>/dev/null | head -n1 || true)" \
      --arg nvcc "$(nvcc --version 2>/dev/null | tail -n1 || true)" \
      '{cargo:$cargo, rustc:$rustc, jq:$jqv, cc:$gcc, nvcc:$nvcc}'
}

get_storage_snapshot() {
    jq -n \
      --arg df "$(df -hT 2>/dev/null || true)" \
      --arg lsblk "$(lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,ROTA 2>/dev/null || true)" \
      '{df_human:$df, lsblk:$lsblk}'
}

get_model_inventory() {
    while IFS='|' read -r name kind rel mode tags; do
        local path="$MODELS_ROOT/$rel"
        local exists=false size_bytes=0 has_manifest=false
        [[ -d "$path" ]] && exists=true
        if [[ "$exists" == "true" ]]; then
            size_bytes="$(du -sb "$path" 2>/dev/null | awk '{print $1}' || echo 0)"
            [[ -f "$path/model.numcert.json" ]] && has_manifest=true
        fi
        jq -n \
          --arg name "$name" \
          --arg kind "$kind" \
          --arg rel "$rel" \
          --arg mode "$mode" \
          --arg tags "$tags" \
          --arg path "$path" \
          --arg exists "$exists" \
          --arg size_bytes "${size_bytes:-0}" \
          --arg has_manifest "$has_manifest" \
          '{
            name:$name,
            kind:$kind,
            relative_path:$rel,
            mode:$mode,
            tags:($tags | split(",")),
            path:$path,
            exists:($exists == "true"),
            size_bytes:($size_bytes | tonumber? // 0),
            has_numcert_manifest:($has_manifest == "true")
          }'
    done < <(case_lines) | jq -s .
}

smoke_lines() {
    cat <<'EOF'
lib_all|cargo test --lib|normal
lib_gguf|cargo test --lib gguf_|normal
tinyllama_q8_0_gguf|cargo test --lib tinyllama_q8_0_gguf -- --nocapture|normal
tinyllama_q4_k_m_gguf|cargo test --lib tinyllama_q4_k_m_gguf -- --nocapture|normal
tinyllama_end_to_end|cargo test --test tinyllama_end_to_end_test --release -- --ignored --nocapture|normal
smollm2_end_to_end|cargo test --test smollm2_end_to_end_test --release -- --ignored --nocapture|normal
qwen25_end_to_end|cargo test --test qwen25_end_to_end_test --release -- --ignored --nocapture|normal
llama32_end_to_end|cargo test --test llama_3_2_end_to_end_test --release -- --ignored --nocapture|normal
EOF

    if [[ "$SUITE" == "full" ]]; then
        cat <<'EOF'
tinyllama_bf16_storage_smoke|cargo test --test tinyllama_bf16_storage_smoke_test --release -- --ignored --nocapture|full
tinyllama_gpu_matmul_smoke|cargo test --test tinyllama_gpu_matmul_smoke_test --release -- --ignored --nocapture|full
tinyllama_disk_spill_smoke|cargo test --test m4_7_4_e_tinyllama_disk_spill_smoke_test --release -- --ignored --nocapture|full
bf16_storage_full_family|cargo test --test bf16_storage_full_family_validation_test --release -- --ignored --nocapture --test-threads=1|full
m47_full_family_validation|cargo test --test m4_7_3_full_family_validation_test --release -- --ignored --nocapture --test-threads=1|full
m85_full_family_validation|cargo test --test m8_5_full_family_validation_test --release -- --ignored --nocapture --test-threads=1|full
m11d_gguf_ignored_diagnostics|cargo test --test tinyllama_f64_validation_test gguf -- --ignored --nocapture --test-threads=1|full
EOF
    fi
}

print_smokes() {
    printf '%-34s %s\n' "name" "command"
    while IFS='|' read -r name command tier; do
        printf '%-34s %s\n' "$name" "$command"
    done < <(smoke_lines)
}

configure_model_test_env() {
    export ATENIA_MODELS_ROOT="$MODELS_ROOT"
    export TINYLLAMA_SAFETENSORS_PATH="${TINYLLAMA_SAFETENSORS_PATH:-$MODELS_ROOT/tinyllama-1.1b/model.safetensors}"
    export SMOLLM2_SAFETENSORS_PATH="${SMOLLM2_SAFETENSORS_PATH:-$MODELS_ROOT/smollm2-1.7b-instruct/model.safetensors}"
    export QWEN25_SAFETENSORS_PATH="${QWEN25_SAFETENSORS_PATH:-$MODELS_ROOT/qwen2.5-1.5b-instruct/model.safetensors}"
    export LLAMA32_SAFETENSORS_PATH="${LLAMA32_SAFETENSORS_PATH:-$MODELS_ROOT/llama-3.2-1b-instruct/model.safetensors}"

    if [[ -f "$MODELS_ROOT/mistral-7b-v0.3/model.safetensors.index.json" ]]; then
        export MISTRAL7B_INDEX_PATH="${MISTRAL7B_INDEX_PATH:-$MODELS_ROOT/mistral-7b-v0.3/model.safetensors.index.json}"
    fi
    if [[ -d "$MODELS_ROOT/llama-2-13b-chat" ]]; then
        export ATENIA_LLAMA2_13B_DIR="${ATENIA_LLAMA2_13B_DIR:-$MODELS_ROOT/llama-2-13b-chat}"
    fi
    if [[ -d "$MODELS_ROOT/tinyllama-1.1b" ]]; then
        export ATENIA_TINYLLAMA_DIR="${ATENIA_TINYLLAMA_DIR:-$MODELS_ROOT/tinyllama-1.1b}"
    fi
}

write_diagnostics() {
    local dir="$1"
    new_dir "$dir"

    nvidia-smi >"$dir/nvidia-smi.txt" 2>&1 || true
    nvidia-smi -q >"$dir/nvidia-smi-query.txt" 2>&1 || true
    df -hT >"$dir/df-hT.txt" 2>&1 || true
    lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,ROTA >"$dir/lsblk.txt" 2>&1 || true
    lscpu >"$dir/lscpu.txt" 2>&1 || true
    uname -a >"$dir/uname.txt" 2>&1 || true
    {
        cargo --version 2>/dev/null || true
        rustc --version 2>/dev/null || true
        jq --version 2>/dev/null || true
        cc --version 2>/dev/null | head -n1 || true
        nvcc --version 2>/dev/null || true
    } >"$dir/tool-versions.txt"
    (cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git status --short) >"$dir/git.txt" 2>&1 || true
    env | sort | grep -E '^(ATENIA|CUDA|CARGO|RUST|LD_LIBRARY_PATH|PATH|MODELS_ROOT|WORK_ROOT|WSL)' >"$dir/env-selected.txt" 2>&1 || true
}

parse_plan() {
    local stderr_path="$1"
    local text manifest mode vt vg rt rg dt dg
    text="$(cat "$stderr_path" 2>/dev/null || true)"

    manifest="$(printf '%s\n' "$text" | sed -nE 's/.*Numeric contract:[[:space:]]*(.*model\.numcert\.json).*/\1/p' | head -n1)"
    mode="$(printf '%s\n' "$text" | sed -nE 's/.*recommended mode:[[:space:]]*([A-Za-z0-9_]+).*/\1/p' | head -n1)"

    read -r vt vg < <(printf '%s\n' "$text" | awk '/^[[:space:]]*VRAM:/ {gsub(/[()]/,""); print $2, $4; exit}')
    read -r rt rg < <(printf '%s\n' "$text" | awk '/^[[:space:]]*RAM:/ {gsub(/[()]/,""); print $2, $4; exit}')
    read -r dt dg < <(printf '%s\n' "$text" | awk '/^[[:space:]]*Disk:/ {gsub(/[()]/,""); print $2, $4; exit}')

    jq -n \
      --arg manifest "$manifest" \
      --arg mode "$mode" \
      --arg vt "${vt:-}" --arg vg "${vg:-}" \
      --arg rt "${rt:-}" --arg rg "${rg:-}" \
      --arg dt "${dt:-}" --arg dg "${dg:-}" \
      'def n: if . == "" then null else tonumber? end;
       {
         manifest_path: (if $manifest == "" then null else $manifest end),
         recommended_mode: (if $mode == "" then null else $mode end),
         vram_tensors: ($vt | n),
         vram_gib: ($vg | n),
         ram_tensors: ($rt | n),
         ram_gib: ($rg | n),
         disk_tensors: ($dt | n),
         disk_gib: ($dg | n)
       }'
}

case_lines() {
    cat <<'EOF'
gguf_tinyllama_q4_quantized|gguf|TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF|manifest|m11d,small
gguf_tinyllama_q8_quantized|gguf|tinyllama-q8_0|manifest|m11d,small
gguf_llama32_1b_q4_quantized|gguf|Llama-3.2-1B-Instruct-Q4_K_M-GGUF|manifest|m11d,1b
gguf_smollm2_q4_quantized|gguf|SmolLM2-1.7B-Instruct-GGUF|manifest|m11d,1b
gguf_phi35_q4_quantized|gguf|Phi-3.5-mini-instruct-Q4_K_M-GGUF|manifest|m11d,phi
st_tinyllama_manifest|safetensors|tinyllama-1.1b|manifest|m46,small
st_tinyllama_fast|safetensors|tinyllama-1.1b|fast|m46,small,fast
st_smollm2_manifest|safetensors|smollm2-1.7b-instruct|manifest|m46,1b
st_smollm2_fast|safetensors|smollm2-1.7b-instruct|fast|m46,1b,fast
st_qwen25_15b_manifest|safetensors|qwen2.5-1.5b-instruct|manifest|m46,1b
st_qwen25_15b_fast|safetensors|qwen2.5-1.5b-instruct|fast|m46,1b,fast
st_llama32_1b_manifest|safetensors|llama-3.2-1b-instruct|manifest|m46,1b
st_llama32_1b_fast|safetensors|llama-3.2-1b-instruct|fast|m46,1b,fast
EOF

    if [[ "$SUITE" == "full" ]]; then
        cat <<'EOF'
st_phi35_manifest|safetensors|phi-3.5-mini-instruct|manifest|top10,phi
st_phi35_fast|safetensors|phi-3.5-mini-instruct|fast|top10,phi,fast
st_mistral7b_manifest|safetensors|mistral-7b-v0.3|manifest|top10,7b
st_mistral7b_fast|safetensors|mistral-7b-v0.3|fast|top10,7b,fast
st_gemma2_2b_manifest|safetensors|gemma-2-2b-it|manifest|top10,2b
st_gemma2_2b_fast|safetensors|gemma-2-2b-it|fast|top10,2b,fast
st_falcon3_7b_manifest|safetensors|falcon3-7b-instruct|manifest|top10,7b
st_falcon3_7b_fast|safetensors|falcon3-7b-instruct|fast|top10,7b,fast
st_llama2_13b_manifest|safetensors|llama-2-13b-chat|manifest|13b,beyond-vram
st_llama2_13b_fast|safetensors|llama-2-13b-chat|fast|13b,beyond-vram,fast
EOF
    fi
}

print_cases() {
    printf '%-34s %-12s %-9s %-78s %s\n' "name" "kind" "mode" "model_path" "exists"
    while IFS='|' read -r name kind rel mode tags; do
        local path="$MODELS_ROOT/$rel"
        local exists="no"
        [[ -d "$path" ]] && exists="yes"
        printf '%-34s %-12s %-9s %-78s %s\n' "$name" "$kind" "$mode" "$path" "$exists"
    done < <(case_lines)
}

csv_escape() {
    local s="${1:-}"
    s="${s//\"/\"\"}"
    printf '"%s"' "$s"
}

run_case() {
    local exe="$1" model_path="$2" mode="$3" stdout_path="$4" stderr_path="$5" case_cache="$6"

    set +e
    if [[ "$mode" == "fast" ]]; then
        (
            export ATENIA_MODELS_ROOT="$MODELS_ROOT"
            export ATENIA_DISK_TIER_DIR="$case_cache"
            if [[ "$CLI_FORCE_BF16" == "true" ]]; then
                export ATENIA_M8_BF16_KERNEL=1
            else
                unset ATENIA_M8_BF16_KERNEL
            fi
            export ATENIA_FAST_MODE=1
            "$exe" generate --prompt "$PROMPT" --model "$model_path" --max-tokens "$MAX_TOKENS" --output json --no-progress
        ) >"$stdout_path" 2>"$stderr_path"
    else
        (
            export ATENIA_MODELS_ROOT="$MODELS_ROOT"
            export ATENIA_DISK_TIER_DIR="$case_cache"
            if [[ "$CLI_FORCE_BF16" == "true" ]]; then
                export ATENIA_M8_BF16_KERNEL=1
            else
                unset ATENIA_M8_BF16_KERNEL
            fi
            unset ATENIA_FAST_MODE
            "$exe" generate --prompt "$PROMPT" --model "$model_path" --max-tokens "$MAX_TOKENS" --output json --no-progress
        ) >"$stdout_path" 2>"$stderr_path"
    fi
    local code=$?
    set -u
    return "$code"
}

run_smoke_command() {
    local name="$1" stdout_path="$2" stderr_path="$3" target_dir="$4" cache_root="$5"

    set +e
    (
        cd "$REPO_ROOT" || exit 1
        export CARGO_TARGET_DIR="$target_dir"
        export ATENIA_MODELS_ROOT="$MODELS_ROOT"
        export ATENIA_TEST_DISK_TIER_BASE="$cache_root/test-cache"
        export ATENIA_DISK_TIER_DIR="$cache_root/smoke-$name"
        unset ATENIA_M8_BF16_KERNEL
        configure_model_test_env
        case "$name" in
            lib_all)
                cargo test --lib
                ;;
            lib_gguf)
                cargo test --lib gguf_
                ;;
            tinyllama_q8_0_gguf)
                cargo test --lib tinyllama_q8_0_gguf -- --nocapture
                ;;
            tinyllama_q4_k_m_gguf)
                cargo test --lib tinyllama_q4_k_m_gguf -- --nocapture
                ;;
            tinyllama_end_to_end)
                cargo test --test tinyllama_end_to_end_test --release -- --ignored --nocapture
                ;;
            smollm2_end_to_end)
                cargo test --test smollm2_end_to_end_test --release -- --ignored --nocapture
                ;;
            qwen25_end_to_end)
                cargo test --test qwen25_end_to_end_test --release -- --ignored --nocapture
                ;;
            llama32_end_to_end)
                cargo test --test llama_3_2_end_to_end_test --release -- --ignored --nocapture
                ;;
            tinyllama_bf16_storage_smoke)
                cargo test --test tinyllama_bf16_storage_smoke_test --release -- --ignored --nocapture
                ;;
            tinyllama_gpu_matmul_smoke)
                cargo test --test tinyllama_gpu_matmul_smoke_test --release -- --ignored --nocapture
                ;;
            tinyllama_disk_spill_smoke)
                cargo test --test m4_7_4_e_tinyllama_disk_spill_smoke_test --release -- --ignored --nocapture
                ;;
            bf16_storage_full_family)
                cargo test --test bf16_storage_full_family_validation_test --release -- --ignored --nocapture --test-threads=1
                ;;
            m47_full_family_validation)
                cargo test --test m4_7_3_full_family_validation_test --release -- --ignored --nocapture --test-threads=1
                ;;
            m85_full_family_validation)
                cargo test --test m8_5_full_family_validation_test --release -- --ignored --nocapture --test-threads=1
                ;;
            m11d_gguf_ignored_diagnostics)
                cargo test --test tinyllama_f64_validation_test gguf -- --ignored --nocapture --test-threads=1
                ;;
            *)
                printf 'unknown smoke: %s\n' "$name" >&2
                exit 2
                ;;
        esac
    ) >"$stdout_path" 2>"$stderr_path" &

    local pid=$!
    local start elapsed last_stdout last_stderr
    start="$(date +%s)"

    while kill -0 "$pid" 2>/dev/null; do
        sleep 10
        elapsed=$(( $(date +%s) - start ))
        last_stdout="$(tail_nonempty_line "$stdout_path")"
        last_stderr="$(tail_nonempty_line "$stderr_path")"
        if [[ -n "$last_stderr" ]]; then
            log_info "[SMOKE $name] running ${elapsed}s | stderr: $last_stderr"
        elif [[ -n "$last_stdout" ]]; then
            log_info "[SMOKE $name] running ${elapsed}s | stdout: $last_stdout"
        else
            log_info "[SMOKE $name] running ${elapsed}s | waiting for first output..."
        fi
    done

    wait "$pid"
    local code=$?
    set -u
    return "$code"
}

run_repo_smokes() {
    local target_dir="$1" cache_root="$2" out_root="$3" raw_root="$4" md="$5"
    local jsonl="$out_root/smokes.jsonl"
    local csv="$out_root/smokes.csv"
    local ok=0 fail=0

    : >"$jsonl"
    printf 'name,tier,exit_code,error,stdout_path,stderr_path\n' >"$csv"

    {
        echo
        echo "## Repo Smokes"
        echo
        echo "| Smoke | Tier | Exit | Error |"
        echo "| --- | --- | ---: | --- |"
    } >>"$md"

    while IFS='|' read -r name command tier; do
        local stdout_path="$raw_root/smoke_$name.stdout.log"
        local stderr_path="$raw_root/smoke_$name.stderr.log"
        local exit_code error status started_at duration tail_line
        started_at="$(date +%s)"

        log_info "[SMOKE] $name: $command"
        log_info "[SMOKE] logs: stdout=$stdout_path stderr=$stderr_path"
        run_smoke_command "$name" "$stdout_path" "$stderr_path" "$target_dir" "$cache_root"
        exit_code=$?
        duration=$(( $(date +%s) - started_at ))
        error=""
        if [[ "$exit_code" -ne 0 ]]; then
            error="$(extract_error_summary "$stderr_path")"
            [[ -n "$error" ]] || error="$(extract_error_summary "$stdout_path")"
            [[ -n "$error" ]] || error="cargo test exited with $exit_code"
            fail=$((fail + 1))
            status="FAIL"
        else
            ok=$((ok + 1))
            status="OK"
        fi

        jq -n \
          --arg name "$name" \
          --arg command "$command" \
          --arg tier "$tier" \
          --arg exit_code "$exit_code" \
          --arg error "$error" \
          --arg stdout_path "$stdout_path" \
          --arg stderr_path "$stderr_path" \
          '{name:$name, command:$command, tier:$tier,
            exit_code:($exit_code | tonumber? // null),
            error:(if $error == "" then null else $error end),
            stdout_path:$stdout_path, stderr_path:$stderr_path}' >>"$jsonl"

        {
            csv_escape "$name"; printf ','
            csv_escape "$tier"; printf ','
            csv_escape "$exit_code"; printf ','
            csv_escape "$error"; printf ','
            csv_escape "$stdout_path"; printf ','
            csv_escape "$stderr_path"; printf '\n'
        } >>"$csv"

        printf '| `%s` | %s | %s | %s |\n' "$name" "$tier" "$exit_code" "$error" >>"$md"
        tail_line="$(tail_nonempty_line "$stdout_path")"
        [[ -n "$tail_line" ]] || tail_line="$(tail_nonempty_line "$stderr_path")"
        if [[ "$status" == "FAIL" ]]; then
            log_info "[SMOKE $status] $name exit=$exit_code duration=${duration}s error=$error"
        else
            log_info "[SMOKE $status] $name exit=$exit_code duration=${duration}s summary=${tail_line:-no output}"
        fi
    done < <(smoke_lines)

    {
        echo
        echo "- Repo smoke OK: $ok"
        echo "- Repo smoke Fail: $fail"
        echo "- Repo smoke JSONL: \`$jsonl\`"
        echo "- Repo smoke CSV: \`$csv\`"
    } >>"$md"
}

main() {
    parse_args "$@"

    [[ "$SUITE" == "quick" || "$SUITE" == "full" ]] || die "SUITE must be quick or full"
    [[ "$BATTERY_MODE" == "smokes" || "$BATTERY_MODE" == "cli" || "$BATTERY_MODE" == "both" ]] || die "BATTERY_MODE must be smokes, cli, or both"

    if [[ "$CLEAN_DEPS" == "true" ]]; then
        clean_generated_artifacts
        clean_dependency_artifacts
        exit 0
    fi

    if [[ "$CLEAN_MODE" == "true" ]]; then
        clean_generated_artifacts
        exit 0
    fi

    resolve_models_root
    [[ -d "$MODELS_ROOT" ]] || die "MODELS_ROOT does not exist: $MODELS_ROOT"
    configure_interactive_run
    [[ "$SUITE" == "quick" || "$SUITE" == "full" ]] || die "SUITE must be quick or full"
    [[ "$BATTERY_MODE" == "smokes" || "$BATTERY_MODE" == "cli" || "$BATTERY_MODE" == "both" ]] || die "BATTERY_MODE must be smokes, cli, or both"

    if [[ "$LIST_ONLY" == "true" ]]; then
        if [[ "$BATTERY_MODE" == "smokes" || "$BATTERY_MODE" == "both" ]]; then
            print_smokes
        fi
        if [[ "$BATTERY_MODE" == "cli" || "$BATTERY_MODE" == "both" ]]; then
            echo
            print_cases
        fi
        exit 0
    fi

    ensure_tools
    configure_model_test_env

    local target_dir="$WORK_ROOT/cargo-target-rtx3090"
    local cache_root="$WORK_ROOT/runtime-cache"
    local run_id="rtx3090_$(date +%Y%m%d_%H%M%S)"
    local out_root="$WORK_ROOT/bench_logs/$run_id"
    local raw_root="$out_root/raw"
    local diag_root="$out_root/diagnostics"
    local jsonl="$out_root/summary.jsonl"
    local csv="$out_root/summary.csv"
    local md="$out_root/summary.md"
    local metadata="$out_root/run_metadata.json"
    local exe="$EXE_PATH"

    new_dir "$target_dir"
    new_dir "$cache_root"
    new_dir "$raw_root"
    new_dir "$diag_root"

    [[ -n "$exe" ]] || exe="$target_dir/release/atenia"

    if [[ "$SKIP_BUILD" != "true" && ( "$BATTERY_MODE" == "cli" || "$BATTERY_MODE" == "both" ) ]]; then
        log_info "Building atenia release binary into $target_dir"
        (cd "$REPO_ROOT" && cargo build --target-dir "$target_dir" --release --bin atenia)
        local build_code=$?
        [[ "$build_code" -eq 0 ]] || die "cargo build failed with exit code $build_code"
        log_success "Build complete"
    fi

    if [[ "$BATTERY_MODE" == "cli" || "$BATTERY_MODE" == "both" ]]; then
        [[ -x "$exe" ]] || die "atenia executable not found or not executable: $exe"
    fi

    local system_json nvidia_json repo_json tools_json storage_json models_json
    system_json="$(json_or_empty_object "$(get_system_snapshot)")"
    nvidia_json="$(json_or_empty_object "$(get_nvidia_snapshot)")"
    repo_json="$(json_or_empty_object "$(get_repo_snapshot)")"
    tools_json="$(json_or_empty_object "$(get_tools_snapshot)")"
    storage_json="$(json_or_empty_object "$(get_storage_snapshot)")"
    models_json="$(get_model_inventory)"
    write_diagnostics "$diag_root"

    jq -n \
      --arg run_id "$run_id" \
      --arg repo_root "$REPO_ROOT" \
      --arg exe_path "$exe" \
      --arg target_dir "$target_dir" \
      --arg cache_root "$cache_root" \
      --arg output_root "$out_root" \
      --arg diagnostics_root "$diag_root" \
      --arg battery_mode "$BATTERY_MODE" \
      --arg system_json "$system_json" \
      --arg nvidia_json "$nvidia_json" \
      --arg repo_json "$repo_json" \
      --arg tools_json "$tools_json" \
      --arg storage_json "$storage_json" \
      --arg models_json "$models_json" \
      'def obj($s): ($s | fromjson? // {});
       def arr($s): ($s | fromjson? // []);
       {
         run_id:$run_id,
         repo_root:$repo_root,
         exe_path:$exe_path,
         target_dir:$target_dir,
         cache_root:$cache_root,
         output_root:$output_root,
         diagnostics_root:$diagnostics_root,
         battery_mode:$battery_mode,
         system:obj($system_json),
         nvidia:obj($nvidia_json),
         repo:obj($repo_json),
         tools:obj($tools_json),
         storage:obj($storage_json),
         model_inventory:arr($models_json)
       }' >"$metadata"

    : >"$jsonl"
    printf 'name,kind,mode,exists,skipped,exit_code,json_parse_ok,tokens_generated,total_seconds,tokens_per_second,eos_reached,vram_tensors,vram_gib,ram_tensors,ram_gib,disk_tensors,disk_gib,recommended_mode,model_path,error\n' >"$csv"

    {
        echo "# Atenia RTX 3090 battery"
        echo
        echo "- Run id: \`$run_id\`"
        echo "- Models root: \`$MODELS_ROOT\`"
        echo "- Work root: \`$WORK_ROOT\`"
        echo "- Diagnostics: \`$diag_root\`"
        echo "- Mode: \`$BATTERY_MODE\`"
        echo "- Suite: \`$SUITE\`"
        echo "- Max tokens: \`$MAX_TOKENS\`"
    } >"$md"

    if [[ "$BATTERY_MODE" == "smokes" || "$BATTERY_MODE" == "both" ]]; then
        run_repo_smokes "$target_dir" "$cache_root" "$out_root" "$raw_root" "$md"
    fi

    local ok=0 fail=0 skip=0

    if [[ "$BATTERY_MODE" == "cli" || "$BATTERY_MODE" == "both" ]]; then
    {
        echo
        echo "## CLI Generation Battery"
        echo
        echo "| Case | Kind | Mode | Exit | JSON | Tokens | tok/s | VRAM | RAM | Disk | Text prefix |"
        echo "| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |"
    } >>"$md"

    while IFS='|' read -r name kind rel mode tags; do
        trace "case start: $name"

        local model_path="$MODELS_ROOT/$rel"
        local stdout_path="$raw_root/$name.stdout.json"
        local stderr_path="$raw_root/$name.stderr.log"
        local env_path="$raw_root/$name.env.json"
        local case_cache="$cache_root/$name"
        local exists=true skipped=false exit_code="" json_parse_ok=false error=""
        local tokens_generated="" total_seconds="" tokens_per_second="" eos_reached="" generated_text_prefix=""

        new_dir "$case_cache"

        if [[ ! -d "$model_path" ]]; then
            exists=false
            skipped=true
            error="model directory missing"
            : >"$stdout_path"
            printf '%s\n' "$error" >"$stderr_path"
        else
            jq -n \
              --arg models "$MODELS_ROOT" \
              --arg disk "$case_cache" \
              --arg bf16 "$( [[ "$CLI_FORCE_BF16" == "true" ]] && echo "1" || echo "" )" \
              --arg fast "$( [[ "$mode" == "fast" ]] && echo "1" || echo "" )" \
              '{ATENIA_MODELS_ROOT:$models, ATENIA_DISK_TIER_DIR:$disk,
                ATENIA_M8_BF16_KERNEL:(if $bf16 == "" then null else $bf16 end),
                ATENIA_FAST_MODE:(if $fast == "" then null else $fast end)}' >"$env_path"

            run_case "$exe" "$model_path" "$mode" "$stdout_path" "$stderr_path" "$case_cache"
            exit_code=$?
            if [[ "$exit_code" -eq 139 ]]; then
                error="segmentation fault"
                printf '\n[rtx3090_battery] process exited with 139 (segmentation fault)\n' >>"$stderr_path"
            elif [[ "$exit_code" -ne 0 ]]; then
                error="$(extract_error_summary "$stderr_path")"
                [[ -n "$error" ]] || error="process exited with $exit_code"
            fi
        fi

        local plan before_gpu after_gpu
        plan="$(json_or_empty_object "$(parse_plan "$stderr_path")")"
        before_gpu="{}"
        after_gpu="$(json_or_empty_object "$(get_nvidia_snapshot)")"

        if [[ -s "$stdout_path" ]] && jq -e . "$stdout_path" >/dev/null 2>&1; then
            json_parse_ok=true
            tokens_generated="$(jq -r '.tokens_generated // empty' "$stdout_path")"
            total_seconds="$(jq -r '.total_seconds // empty' "$stdout_path")"
            tokens_per_second="$(jq -r '.tokens_per_second // empty' "$stdout_path")"
            eos_reached="$(jq -r '.eos_reached // empty' "$stdout_path")"
            generated_text_prefix="$(jq -r '.generated_text // ""' "$stdout_path" | tr '\r\n|' '   ' | cut -c1-160)"
        elif [[ "$skipped" != "true" && -z "$error" ]]; then
            error="json parse failed"
        fi

        local result
        result="$(jq -n \
          --arg run_id "$run_id" \
          --arg name "$name" \
          --arg kind "$kind" \
          --arg mode "$mode" \
          --arg tags "$tags" \
          --arg model_path "$model_path" \
          --arg exists "$exists" \
          --arg skipped "$skipped" \
          --arg exit_code "$exit_code" \
          --arg json_parse_ok "$json_parse_ok" \
          --arg tokens_generated "$tokens_generated" \
          --arg total_seconds "$total_seconds" \
          --arg tokens_per_second "$tokens_per_second" \
          --arg eos_reached "$eos_reached" \
          --arg text "$generated_text_prefix" \
          --arg stdout_path "$stdout_path" \
          --arg stderr_path "$stderr_path" \
          --arg env_path "$env_path" \
          --arg cache_dir "$case_cache" \
          --arg error "$error" \
          --arg plan_json "$plan" \
          --arg before_gpu_json "$before_gpu" \
          --arg after_gpu_json "$after_gpu" \
          'def b: . == "true";
           def n: if . == "" then null else tonumber? end;
           def obj($s): ($s | fromjson? // {});
           (obj($plan_json)) as $plan
           | (obj($before_gpu_json)) as $before_gpu
           | (obj($after_gpu_json)) as $after_gpu
           | {
             run_id:$run_id,
             name:$name,
             kind:$kind,
             mode:$mode,
             tags:($tags | split(",")),
             model_path:$model_path,
             exists:($exists | b),
             skipped:($skipped | b),
             exit_code:($exit_code | n),
             json_parse_ok:($json_parse_ok | b),
             tokens_generated:($tokens_generated | n),
             total_seconds:($total_seconds | n),
             tokens_per_second:($tokens_per_second | n),
             eos_reached:(if $eos_reached == "" then null else ($eos_reached | b) end),
             generated_text_prefix:$text,
             stdout_path:$stdout_path,
             stderr_path:$stderr_path,
             env_path:$env_path,
             cache_dir:$cache_dir,
             plan:$plan,
             before_gpu:($before_gpu.gpu_query // []),
             after_gpu:($after_gpu.gpu_query // []),
             error:(if $error == "" then null else $error end)
           }')"
        result="$(json_or_empty_object "$result")"
        printf '%s\n' "$result" >>"$jsonl"

        local vram_tensors vram_gib ram_tensors ram_gib disk_tensors disk_gib rec_mode
        vram_tensors="$(jq -r '.vram_tensors // ""' <<<"$plan")"
        vram_gib="$(jq -r '.vram_gib // ""' <<<"$plan")"
        ram_tensors="$(jq -r '.ram_tensors // ""' <<<"$plan")"
        ram_gib="$(jq -r '.ram_gib // ""' <<<"$plan")"
        disk_tensors="$(jq -r '.disk_tensors // ""' <<<"$plan")"
        disk_gib="$(jq -r '.disk_gib // ""' <<<"$plan")"
        rec_mode="$(jq -r '.recommended_mode // ""' <<<"$plan")"

        {
            csv_escape "$name"; printf ','
            csv_escape "$kind"; printf ','
            csv_escape "$mode"; printf ','
            csv_escape "$exists"; printf ','
            csv_escape "$skipped"; printf ','
            csv_escape "$exit_code"; printf ','
            csv_escape "$json_parse_ok"; printf ','
            csv_escape "$tokens_generated"; printf ','
            csv_escape "$total_seconds"; printf ','
            csv_escape "$tokens_per_second"; printf ','
            csv_escape "$eos_reached"; printf ','
            csv_escape "$vram_tensors"; printf ','
            csv_escape "$vram_gib"; printf ','
            csv_escape "$ram_tensors"; printf ','
            csv_escape "$ram_gib"; printf ','
            csv_escape "$disk_tensors"; printf ','
            csv_escape "$disk_gib"; printf ','
            csv_escape "$rec_mode"; printf ','
            csv_escape "$model_path"; printf ','
            csv_escape "$error"; printf '\n'
        } >>"$csv"

        local status="FAIL"
        if [[ "$skipped" == "true" ]]; then
            status="SKIP"
            skip=$((skip + 1))
        elif [[ "$exit_code" == "0" && "$json_parse_ok" == "true" ]]; then
            status="OK"
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi

        local tps_cell="" vram_cell="" ram_cell="" disk_cell=""
        [[ -n "$tokens_per_second" ]] && tps_cell="$(printf '%.3f' "$tokens_per_second" 2>/dev/null || printf '%s' "$tokens_per_second")"
        [[ -n "$vram_tensors" ]] && vram_cell="$vram_tensors / $vram_gib GiB"
        [[ -n "$ram_tensors" ]] && ram_cell="$ram_tensors / $ram_gib GiB"
        [[ -n "$disk_tensors" ]] && disk_cell="$disk_tensors / $disk_gib GiB"

        printf '| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
          "$name" "$kind" "$mode" "${exit_code:-}" "$json_parse_ok" "${tokens_generated:-}" "$tps_cell" \
          "$vram_cell" "$ram_cell" "$disk_cell" "$generated_text_prefix" >>"$md"

        if [[ "$status" == "FAIL" && -n "$error" ]]; then
            log_info "[$status] $name exit=${exit_code:-} json=$json_parse_ok tok=${tokens_generated:-} tps=${tps_cell:-} error=$error"
        else
            log_info "[$status] $name exit=${exit_code:-} json=$json_parse_ok tok=${tokens_generated:-} tps=${tps_cell:-}"
        fi
    done < <(case_lines)
    fi

    {
        echo
        echo "## Totals"
        echo
        if [[ "$BATTERY_MODE" == "cli" || "$BATTERY_MODE" == "both" ]]; then
            echo "- CLI OK: $ok"
            echo "- CLI Fail: $fail"
            echo "- CLI Skip: $skip"
        fi
        echo
        echo "## Files"
        echo
        echo "- Metadata: \`$metadata\`"
        echo "- JSONL: \`$jsonl\`"
        echo "- CSV: \`$csv\`"
        echo "- Raw logs: \`$raw_root\`"
    } >>"$md"

    log_success "Battery complete"
    log_info "Summary: $md"
    log_info "CSV: $csv"
    log_info "JSONL: $jsonl"
}

main "$@"
