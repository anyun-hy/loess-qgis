from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "bash" / "ssh_tencent.sh"


def test_wrapper_reuses_tencent_connection_with_user_owned_socket(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "ssh-arguments"
    fake_ssh = bin_dir / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\0' \"$@\" > \"${SSH_CAPTURE}\"\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(fake_ssh.stat().st_mode | stat.S_IXUSR)

    control_dir = tmp_path / "control"
    env = dict(os.environ)
    env.update(
        {
            "LOESS_SSH_CONTROL_DIR": str(control_dir),
            "LOESS_SSH_HOST": "ubuntu-validation",
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "SSH_CAPTURE": str(capture),
        }
    )
    result = subprocess.run(
        [str(WRAPPER), "printf '%s\\n' ready"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_bytes().split(b"\0")[:-1] == [
        b"-o",
        b"ControlMaster=auto",
        b"-o",
        b"ControlPersist=15m",
        b"-o",
        f"ControlPath={control_dir}/%C".encode(),
        b"ubuntu-validation",
        b"printf '%s\\n' ready",
    ]
    assert stat.S_IMODE(control_dir.stat().st_mode) == 0o700
