from pathlib import Path
import base64

root = Path(__file__).resolve().parent.parent
raw = (root / "grab_it_il.tgz").read_bytes()
b64 = base64.b64encode(raw).decode("ascii")
wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76))
out = root / "scripts" / "install_grab_it_il_embed.py"
out.write_text(
    "#!/usr/bin/env python3\n"
    '"""Extract grab_it_il into /workspaces/isaac_ros-dev/src (no network)."""\n'
    "from __future__ import annotations\n\n"
    "import base64\n"
    "import io\n"
    "import tarfile\n"
    "from pathlib import Path\n\n"
    'DEST = Path("/workspaces/isaac_ros-dev/src")\n'
    'TGZ_B64 = """\n'
    + wrapped
    + '\n"""\n\n\n'
    "def main() -> None:\n"
    "    DEST.mkdir(parents=True, exist_ok=True)\n"
    '    raw = base64.b64decode("".join(TGZ_B64.split()))\n'
    '    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:\n'
    "        tf.extractall(DEST)\n"
    '    pkg = DEST / "grab_it_il" / "package.xml"\n'
    "    if not pkg.is_file():\n"
    '        raise SystemExit(f"extract failed: {pkg}")\n'
    '    print(f"ok {pkg}")\n\n\n'
    'if __name__ == "__main__":\n'
    "    main()\n",
    encoding="utf-8",
)
print("wrote", out, out.stat().st_size)
