from pathlib import Path

p = Path(r"C:\Users\chach\Projects\grab_it-dev-ws\scripts\_il_wrist_finish.py")
lines = p.read_text(encoding="utf-8").splitlines()
cut = next(i for i, line in enumerate(lines) if line.startswith("def _dead_servo_removed"))
keep = "\n".join(lines[:cut]).rstrip() + "\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n"
p.write_text(keep, encoding="utf-8")
print("trimmed at", cut)
