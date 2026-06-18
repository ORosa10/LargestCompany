from __future__ import annotations

from pathlib import Path


APP_CORE_PATH = Path(__file__).with_name("app_core.py")
source = APP_CORE_PATH.read_text(encoding="utf-8")
exec(compile(source, str(APP_CORE_PATH), "exec"), globals())
