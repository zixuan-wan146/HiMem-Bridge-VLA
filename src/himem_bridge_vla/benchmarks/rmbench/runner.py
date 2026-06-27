from __future__ import annotations

import os
import subprocess

from himem_bridge_vla.path_utils import find_repo_root


def main() -> int:
    repo_root = find_repo_root(__file__)
    src_root = repo_root / "src"
    script = repo_root / "scripts" / "eval" / "run_rmbench_eval.sh"
    env = dict(os.environ)
    pythonpath = [str(src_root), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return subprocess.call(["bash", str(script)], cwd=repo_root, env=env)
