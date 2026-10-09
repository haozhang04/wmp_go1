#!/usr/bin/env bash

set -e

export LEGGEDSKILL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
LEGGED_GYM_SCRIPTS_PATH="${LEGGEDSKILL_PATH}/legged_gym/scripts"
DEFAULT_TASK="a1_amp"

if [[ -n "${CONDA_PREFIX}" ]]; then
    python_exe="${CONDA_PREFIX}/bin/python"
else
    echo "[Error] No conda environment activated. Please activate the conda environment first."
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

_leggedskill_usage() {
    echo "Usage:"
    echo "  ./bash/leggedskill.sh -t --task a1"
    echo "  ./bash/leggedskill.sh -p --task a1"
    echo "  ./bash/leggedskill.sh -v --task a1"
    echo ""
    echo "Options:"
    echo "  -t, --train     Train with legged_gym scripts/train.py, headless by default"
    echo "  -p, --play      Play with legged_gym scripts/play.py"
    echo "  -v, --terrain-view  Visualize the full training terrain map"
    echo ""
    echo "Default task: ${DEFAULT_TASK}"
}

_leggedskill_has_task_arg() {
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "--task" || "${arg}" == --task=* ]]; then
            return 0
        fi
    done
    return 1
}

_leggedskill_run_python() {
    local script="$1"
    shift
    local args=("$@")
    if ! _leggedskill_has_task_arg "${args[@]}"; then
        args=(--task "${DEFAULT_TASK}" "${args[@]}")
    fi
    PYTHONPATH="${LEGGEDSKILL_PATH}:${PYTHONPATH}" \
        "${python_exe}" "${script}" "${args[@]}"
}

_leggedskill_python_argcomplete_wrapper() {
    local IFS=$'\013'
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    PYTHONPATH="${LEGGEDSKILL_PATH}:${PYTHONPATH}" \
                    "${python_exe}" "${LEGGED_GYM_SCRIPTS_PATH}/train.py" 8>&1 9>&2 1>/dev/null 2>/dev/null) )
}

complete -o nospace -F _leggedskill_python_argcomplete_wrapper "./bash/leggedskill.sh" 2> /dev/null || true
complete -o nospace -F _leggedskill_python_argcomplete_wrapper "./leggedskill.sh" 2> /dev/null || true

case "$1" in
    -t|--train)
        shift
        _leggedskill_run_python "${LEGGED_GYM_SCRIPTS_PATH}/train.py" --headless "$@"
        ;;
    -p|--play)
        shift
        _leggedskill_run_python "${LEGGED_GYM_SCRIPTS_PATH}/play.py" "$@"
        ;;
    -v|--terrain-view)
        shift
        _leggedskill_run_python "${LEGGED_GYM_SCRIPTS_PATH}/view_terrain.py" "$@"
        ;;
    "")
        _leggedskill_usage
        ;;
    *)
        echo "[Error] Unknown option: $1"
        _leggedskill_usage
        exit 1
        ;;
esac
