#!/usr/bin/env bash

libero_profile_allowed_key() {
  case "$1" in
    HIMEM_SERVER_URI | \
      HIMEM_MUJOCO_GL | \
      HIMEM_LIBERO_EPISODES | \
      HIMEM_LIBERO_TASK_SUITES | \
      HIMEM_LIBERO_TASK_LIMIT | \
      HIMEM_LIBERO_MAX_STEPS | \
      HIMEM_LIBERO_HORIZON | \
      HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT | \
      HIMEM_LIBERO_TRANSITION_DATASET_NAME | \
      HIMEM_LIBERO_TRANSITION_TRACE_FILE | \
      HIMEM_LIBERO_CKPT_NAME | \
      HIMEM_LIBERO_RUN_DIR | \
      HIMEM_LIBERO_LOG_DIR | \
      HIMEM_LIBERO_VIDEO_DIR | \
      HIMEM_LIBERO_LOG_FILE | \
      HIMEM_LIBERO_RESULT_FILE | \
      HIMEM_LIBERO_MANIFEST_FILE | \
      LIBERO_PYTHON | \
      PYOPENGL_PLATFORM)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

load_libero_profile() {
  local repo_root=$1
  local profile=${HIMEM_LIBERO_PROFILE:-}
  local profile_path
  local line key value

  if [ -z "$profile" ]; then
    return 0
  fi

  case "$profile" in
    /*)
      printf '[libero-profile] ERROR: HIMEM_LIBERO_PROFILE must be project-relative: %s\n' "$profile" >&2
      return 1
      ;;
  esac
  profile_path="$repo_root/$profile"

  if [ ! -f "$profile_path" ]; then
    printf '[libero-profile] ERROR: profile file does not exist: %s\n' "$profile" >&2
    return 1
  fi

  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      "" | "#"*) continue ;;
    esac
    case "$line" in
      *=*) ;;
      *)
        printf '[libero-profile] ERROR: invalid line in %s: %s\n' "$profile" "$line" >&2
        return 1
        ;;
    esac

    key=${line%%=*}
    value=${line#*=}
    if ! libero_profile_allowed_key "$key"; then
      printf '[libero-profile] ERROR: unsupported key in %s: %s\n' "$profile" "$key" >&2
      return 1
    fi
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$profile_path"

  export HIMEM_LIBERO_PROFILE="$profile"
}
