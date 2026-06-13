#!/usr/bin/env bash

calvin_profile_allowed_key() {
  case "$1" in
    HIMEM_SERVER_URI | \
      HIMEM_MUJOCO_GL | \
      HIMEM_CALVIN_ROOT | \
      HIMEM_CALVIN_DATASET_PATH | \
      HIMEM_CALVIN_ANNOTATIONS_PATH | \
      HIMEM_CALVIN_NUM_SEQUENCES | \
      HIMEM_CALVIN_SEQUENCE_OFFSET | \
      HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK | \
      HIMEM_CALVIN_HORIZON | \
      HIMEM_CALVIN_CKPT_NAME | \
      HIMEM_CALVIN_RUN_DIR | \
      HIMEM_CALVIN_LOG_DIR | \
      HIMEM_CALVIN_VIDEO_DIR | \
      HIMEM_CALVIN_LOG_FILE | \
      HIMEM_CALVIN_RESULT_FILE | \
      HIMEM_CALVIN_MANIFEST_FILE | \
      HIMEM_CALVIN_SAVE_VIDEO | \
      HIMEM_CALVIN_VIDEO_FPS | \
      HIMEM_CALVIN_SEED | \
      HIMEM_CALVIN_GRIPPER_MODE | \
      HIMEM_CALVIN_RESET_MEMORY_SCOPE | \
      HIMEM_CALVIN_SHOW_GUI | \
      CALVIN_PYTHON | \
      PYOPENGL_PLATFORM)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

load_calvin_profile() {
  local repo_root=$1
  local profile=${HIMEM_CALVIN_PROFILE:-}
  local profile_path
  local line key value

  if [ -z "$profile" ]; then
    return 0
  fi

  case "$profile" in
    /*)
      printf '[calvin-profile] ERROR: HIMEM_CALVIN_PROFILE must be project-relative: %s\n' "$profile" >&2
      return 1
      ;;
  esac
  profile_path="$repo_root/$profile"

  if [ ! -f "$profile_path" ]; then
    printf '[calvin-profile] ERROR: profile file does not exist: %s\n' "$profile" >&2
    return 1
  fi

  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      "" | "#"*) continue ;;
    esac
    case "$line" in
      *=*) ;;
      *)
        printf '[calvin-profile] ERROR: invalid line in %s: %s\n' "$profile" "$line" >&2
        return 1
        ;;
    esac

    key=${line%%=*}
    value=${line#*=}
    if ! calvin_profile_allowed_key "$key"; then
      printf '[calvin-profile] ERROR: unsupported key in %s: %s\n' "$profile" "$key" >&2
      return 1
    fi
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$profile_path"

  export HIMEM_CALVIN_PROFILE="$profile"
}
