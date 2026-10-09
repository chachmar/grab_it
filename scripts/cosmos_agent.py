#!/usr/bin/env python3
"""Cosmos3-Edge reasoner → grab_it GUI WebSocket agent.

Plans with nvidia/Cosmos3-Edge, then executes via the same WS API as gui.html
(navigate named places, dock/undock, grasp click/execute, etc.).

Examples (AGX host, nav + GUI already up):
  source ~/cosmos3-venv/bin/activate
  pip install websockets pillow requests   # once
  python scripts/cosmos_agent.py --no-camera "go to the kitchen"
  python scripts/cosmos_agent.py --dry-run --no-camera "go to the kitchen"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
import time
from typing import Any

DEFAULT_WS = os.environ.get("GRAB_IT_WS_URL", "ws://127.0.0.1:8766")
DEFAULT_CAMERA = os.environ.get("GRAB_IT_CAMERA_URL", "http://127.0.0.1:8081/d435i")
DEFAULT_LOCATIONS = ("kitchen", "living_room", "trash", "dock_area")
MODEL_ID = os.environ.get("COSMOS3_MODEL", "nvidia/Cosmos3-Edge")

ALLOWED_OPS = frozenset(
    {
        "navigate",
        "cancel_nav",
        "dock",
        "undock",
        "home",
        "sleep",
        "grasp_click",
        "grasp_execute",
        "grasp_cancel",
        "wait",
        "say",
    }
)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Prefer the last JSON object (after any thinking chatter).
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _repair_json(text: str) -> dict[str, Any] | None:
    """Best-effort salvage when generation is cut off mid-object."""
    start = text.find("{")
    if start < 0:
        return None
    chunk = text[start:]
    # If a complete object exists, use the normal parser.
    try:
        return _extract_json(chunk)
    except Exception:
        pass
    # Truncated: close open braces/brackets and retry.
    open_curly = chunk.count("{") - chunk.count("}")
    open_square = chunk.count("[") - chunk.count("]")
    if open_curly < 0 or open_square < 0:
        return None
    # Trim to last safe comma / colon boundary-ish
    trimmed = chunk.rstrip()
    if trimmed.endswith(","):
        trimmed = trimmed[:-1]
    repaired = trimmed + ("]" * open_square) + ("}" * open_curly)
    try:
        return json.loads(repaired)
    except Exception:
        return None


def try_fast_plan(user_text: str, locations: list[str]) -> dict[str, Any] | None:
    """Skip the LLM for common one-shot commands (fast + reliable on Orin)."""
    t = user_text.lower().strip()
    if re.search(r"\b(cancel|stop)\b.*\b(nav|navigation|going)\b", t) or t in (
        "cancel",
        "stop",
        "cancel nav",
        "cancel navigation",
    ):
        return {"say": "Canceling navigation.", "steps": [{"op": "cancel_nav"}]}
    if re.search(r"\bundock\b", t):
        return {"say": "Undocking.", "steps": [{"op": "undock"}]}
    if re.search(r"\bdock\b", t) and "undock" not in t:
        return {"say": "Docking.", "steps": [{"op": "dock"}]}
    if re.search(r"\b(go\s+)?home\b", t) and "living" not in t:
        return {"say": "Arm to home.", "steps": [{"op": "home"}]}
    if re.search(r"\bsleep\b", t):
        return {"say": "Arm to sleep.", "steps": [{"op": "sleep"}]}

    nav_intent = bool(
        re.search(r"\b(go|navigate|drive|move|take me|head)\b", t)
        or re.search(r"\bto the\b", t)
    )
    if not nav_intent:
        return None

    # Prefer longer names first (living_room before room).
    ranked = sorted(locations, key=len, reverse=True)
    for name in ranked:
        human = name.replace("_", " ")
        if name in t or human in t:
            return {
                "say": f"Navigating to {human}.",
                "steps": [{"op": "navigate", "name": name}],
            }
    return None


def fetch_camera_jpeg(url: str, timeout: float = 5.0) -> bytes | None:
    try:
        import requests
    except ImportError:
        print("[agent] install requests for --camera support: pip install requests")
        return None
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "multipart" in ctype:
                buf = b""
                for chunk in r.iter_content(chunk_size=4096):
                    if not chunk:
                        break
                    buf += chunk
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                    if a >= 0 and b >= 0:
                        return buf[a : b + 2]
                    if len(buf) > 8_000_000:
                        break
                return None
            data = r.content
            if data[:2] == b"\xff\xd8":
                return data
            a = data.find(b"\xff\xd8")
            b = data.find(b"\xff\xd9", a + 2) if a >= 0 else -1
            if a >= 0 and b >= 0:
                return data[a : b + 2]
            return None
    except Exception as e:
        print(f"[agent] camera fetch failed ({url}): {e}")
        return None


class CosmosPlanner:
    def __init__(self, model_id: str = MODEL_ID, lazy: bool = True):
        self.model_id = model_id
        self._torch = None
        self.processor = None
        self.model = None
        if not lazy:
            self._ensure_loaded()

    def _ensure_loaded(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        print(f"[agent] loading {self.model_id} …")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        print("[agent] model ready")

    def plan(
        self,
        user_text: str,
        locations: list[str],
        image_jpeg: bytes | None = None,
        max_new_tokens: int = 160,
    ) -> dict[str, Any]:
        self._ensure_loaded()
        locs = ", ".join(locations)
        # Few-shot + hard ban on chain-of-thought — Cosmos loves to narrate otherwise.
        system = (
            "You are a robot command compiler. Respond with ONE JSON object only.\n"
            "Forbidden: explanations, analysis, markdown, <think> tags, or text before '{'.\n"
            f"Valid place names: {locs}\n"
            "Valid ops: navigate, cancel_nav, dock, undock, home, sleep, "
            "grasp_click (needs u,v in 0..1), grasp_execute, grasp_cancel, wait.\n"
            "Example user: go to the kitchen\n"
            'Example assistant: {"say":"Navigating to the kitchen.","steps":[{"op":"navigate","name":"kitchen"}]}\n'
            "Example user: cancel navigation\n"
            'Example assistant: {"say":"Canceling navigation.","steps":[{"op":"cancel_nav"}]}\n'
            f"User: {user_text}\n"
            "Assistant:"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": system}]
        tmp_path = None
        if image_jpeg:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(image_jpeg)
            content.insert(0, {"type": "image", "path": tmp_path})

        messages = [{"role": "user", "content": content}]
        text = ""
        try:
            print("[agent] tokenizing…")
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Prefill "{" so the model continues a JSON object instead of prose.
            brace = self.processor.tokenizer("{", add_special_tokens=False).input_ids
            if not isinstance(brace, list):
                brace = brace.tolist()
            import torch

            prefill = torch.tensor([brace], device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], prefill], dim=1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    [
                        inputs["attention_mask"],
                        torch.ones_like(prefill),
                    ],
                    dim=1,
                )

            print(f"[agent] generating (max_new_tokens={max_new_tokens}, JSON prefill)…")
            t0 = time.monotonic()
            with self._torch.inference_mode():
                try:
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        stop_strings=["\nOkay", "\nFirst", "\nLet me", "\nThe "],
                        tokenizer=self.processor.tokenizer,
                    )
                except TypeError:
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
            print(f"[agent] generate done in {time.monotonic() - t0:.1f}s")
            # Decode from the prefaced "{" through new tokens.
            start = inputs["input_ids"].shape[1] - len(brace)
            text = self.processor.tokenizer.decode(
                out[0, start:], skip_special_tokens=True
            )
            if not text.lstrip().startswith("{"):
                text = "{" + text
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        print(f"[agent] raw model output:\n{text}\n")
        try:
            return _extract_json(text)
        except Exception as e:
            # Incomplete JSON is common when the model rambling-cuts mid-object.
            repaired = _repair_json(text)
            if repaired is not None:
                print("[agent] repaired partial JSON")
                return repaired
            return {"say": f"Could not parse plan ({e})", "steps": [], "raw": text}


class GrabItClient:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._ws = None
        self._reader_task: asyncio.Task | None = None
        self.locations: list[str] = list(DEFAULT_LOCATIONS)
        self.last_nav_status = ""
        self.last_grasp_status: dict[str, Any] = {}

    async def connect(self):
        import websockets

        self._ws = await websockets.connect(self.ws_url, max_size=8_000_000)
        print(f"[agent] connected to {self.ws_url}")
        self._reader_task = asyncio.create_task(self._reader_loop(), name="ws-reader")
        # Let hello / locations arrive.
        await asyncio.sleep(0.5)

    async def _reader_loop(self):
        assert self._ws is not None
        try:
            async for raw in self._ws:
                self._handle_msg(raw)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[agent] ws reader stopped: {e}")

    def _handle_msg(self, raw: str | bytes):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type")
        if t == "locations":
            names = msg.get("names") or []
            if names:
                self.locations = list(names)
                print(f"[agent] locations from GUI: {self.locations}")
        elif t == "nav_status":
            self.last_nav_status = msg.get("text", "")
            print(f"[nav] {self.last_nav_status}")
        elif t == "grasp_status":
            self.last_grasp_status = msg
            print(f"[grasp] {msg}")
        # Ignore high-rate joint_state / servo_health spam.

    async def send(self, payload: dict[str, Any]):
        assert self._ws is not None
        await self._ws.send(json.dumps(payload))
        print(f"[agent] sent {payload}")

    async def pump(self, seconds: float):
        await asyncio.sleep(seconds)

    async def close(self):
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


async def execute_plan(
    client: GrabItClient,
    plan: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    say = plan.get("say")
    if say:
        print(f"[say] {say}")
    steps = plan.get("steps") or []
    if not steps:
        print("[agent] no steps to run")
        return

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            print(f"[agent] skip bad step {step!r}")
            continue
        op = step.get("op")
        print(f"[agent] step {i + 1}/{len(steps)}: {step}")
        if op not in ALLOWED_OPS:
            print(f"[agent] unknown op {op!r}, skipping")
            continue
        if dry_run:
            continue

        if op == "say":
            print(f"[say] {step.get('text', '')}")
        elif op == "wait":
            await client.pump(float(step.get("seconds", 1)))
        elif op == "navigate":
            name = step.get("name", "")
            if name not in client.locations:
                print(f"[agent] unknown place {name!r}; known={client.locations}")
                continue
            await client.send({"type": "navigate", "name": name})
            await client.pump(2.0)
        elif op == "cancel_nav":
            await client.send({"type": "cancel_nav"})
            await client.pump(0.5)
        elif op == "dock":
            await client.send({"type": "dock"})
            await client.pump(1.0)
        elif op == "undock":
            await client.send({"type": "undock"})
            await client.pump(1.0)
        elif op == "home":
            await client.send({"type": "home"})
            await client.pump(0.5)
        elif op == "sleep":
            await client.send({"type": "sleep"})
            await client.pump(0.5)
        elif op == "grasp_click":
            await client.send(
                {
                    "type": "grasp_click",
                    "u": float(step["u"]),
                    "v": float(step["v"]),
                }
            )
            await client.pump(1.5)
        elif op == "grasp_execute":
            await client.send({"type": "grasp_execute"})
            await client.pump(2.0)
        elif op == "grasp_cancel":
            await client.send({"type": "grasp_cancel"})
            await client.pump(0.5)


async def handle_command(
    planner: CosmosPlanner | None,
    client: GrabItClient,
    text: str,
    *,
    camera_url: str | None,
    use_camera: bool,
    dry_run: bool,
    force_llm: bool,
) -> None:
    plan = None if force_llm else try_fast_plan(text, client.locations)
    if plan is not None:
        print("[agent] fast-path (no LLM)")
    else:
        if planner is None:
            print("[agent] no LLM loaded and no fast-path match")
            return
        image = None
        if use_camera and camera_url:
            print(f"[agent] grabbing frame from {camera_url}")
            image = await asyncio.to_thread(fetch_camera_jpeg, camera_url)
            if image:
                print(f"[agent] got JPEG ({len(image)} bytes)")
            else:
                print("[agent] no camera frame; planning text-only")

        async def _heartbeat():
            n = 0
            while True:
                await asyncio.sleep(5)
                n += 5
                print(f"[agent] still planning… ({n}s)")

        hb = asyncio.create_task(_heartbeat())
        try:
            plan = await asyncio.to_thread(
                planner.plan, text, client.locations, image
            )
        finally:
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass

        # If the LLM babbled and produced no steps, fall back to fast-path.
        if not (plan.get("steps") or []):
            fb = try_fast_plan(text, client.locations)
            if fb is not None:
                print("[agent] LLM plan empty/unparsed — using fast-path fallback")
                plan = fb

    print(f"[agent] plan: {json.dumps(plan, indent=2)}")
    await execute_plan(client, plan, dry_run=dry_run)


async def amain(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cosmos3-Edge → grab_it WebSocket agent")
    p.add_argument("command", nargs="?", help="Single command; omit for interactive REPL")
    p.add_argument("--ws", default=DEFAULT_WS, help="GUI WebSocket URL")
    p.add_argument("--camera", default=DEFAULT_CAMERA, help="MJPEG/JPEG camera URL")
    p.add_argument("--no-camera", action="store_true", help="Never attach a camera frame")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not send WS commands")
    p.add_argument("--model", default=MODEL_ID, help="HF model id")
    p.add_argument(
        "--llm",
        action="store_true",
        help="Force Cosmos LLM even when a fast-path match exists",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Never load Cosmos; only fast-path commands (navigate/dock/…)",
    )
    args = p.parse_args(argv)

    planner = None if args.no_llm else CosmosPlanner(args.model, lazy=True)
    client = GrabItClient(args.ws)
    try:
        if not args.dry_run:
            await client.connect()
        else:
            print("[agent] dry-run: skipping WebSocket connect")

        use_cam = not args.no_camera
        if args.command:
            await handle_command(
                planner,
                client,
                args.command,
                camera_url=args.camera,
                use_camera=use_cam,
                dry_run=args.dry_run,
                force_llm=args.llm,
            )
            return 0

        print("Interactive mode. Fast-path handles go-to / dock / cancel without LLM.")
        print("Use --llm or type /llm on for full Cosmos planning.")
        print("Commands: /camera on|off  /dry on|off  /llm on|off  /quit")
        dry = args.dry_run
        force_llm = args.llm
        while True:
            try:
                line = input("cosmos> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/q", "/quit", "quit", "exit"):
                break
            if line.startswith("/camera"):
                parts = line.split()
                use_cam = not (len(parts) > 1 and parts[1] in ("off", "0", "false"))
                print(f"[agent] camera={'on' if use_cam else 'off'}")
                continue
            if line.startswith("/dry"):
                parts = line.split()
                dry = not (len(parts) > 1 and parts[1] in ("off", "0", "false"))
                print(f"[agent] dry-run={dry}")
                continue
            if line.startswith("/llm"):
                parts = line.split()
                force_llm = not (len(parts) > 1 and parts[1] in ("off", "0", "false"))
                print(f"[agent] force_llm={force_llm}")
                continue
            await handle_command(
                planner,
                client,
                line,
                camera_url=args.camera,
                use_camera=use_cam,
                dry_run=dry,
                force_llm=force_llm,
            )
        return 0
    finally:
        await client.close()


def main():
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
