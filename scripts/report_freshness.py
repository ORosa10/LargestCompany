"""Guard the headless report pipeline against stale and duplicate runs.

The price/IV inputs are still supplied manually in daily_inputs.json. A push to
that file remains the fastest trigger; scheduled runs are a safe fallback that
only process inputs marked for today and only at the three local checkpoints
used by the former cloud task.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
INPUTS_PATH = REPO / "daily_inputs.json"
REPORT_PATH = REPO / "reports" / "latest.md"
LOCAL_TZ = ZoneInfo("Europe/Prague")
LOCAL_SLOTS = {(9, 30), (12, 30), (17, 0)}
REPORT_DATE_RE = re.compile(
    r"^# LargestCompany daily report - (?P<date>\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)


def _set_output(name: str, value: str) -> None:
    print(f"{name}={value}")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def _parse_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(LOCAL_TZ)


def _git_timestamp(path: str) -> datetime | None:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", path],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        timestamp = int(result.stdout.strip())
    except ValueError:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(LOCAL_TZ)


def _existing_report_date() -> str | None:
    if not REPORT_PATH.exists():
        return None
    match = REPORT_DATE_RE.search(REPORT_PATH.read_text(encoding="utf-8"))
    return match.group("date") if match else None


def _near_local_checkpoint(now: datetime) -> bool:
    current_minutes = now.hour * 60 + now.minute
    return any(
        abs(current_minutes - (hour * 60 + minute)) <= 20
        for hour, minute in LOCAL_SLOTS
    )


def main() -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "local")
    force = os.environ.get("FORCE_REPORT", "").lower() == "true"
    now = datetime.now(LOCAL_TZ)
    today = now.date().isoformat()

    try:
        inputs = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _set_output("run_report", "false")
        _set_output("report_date", today)
        _set_output("reason", f"daily_inputs.json nelze načíst: {exc}")
        return 0

    trigger = _parse_datetime(inputs.get("_trigger"))
    input_commit = _git_timestamp("daily_inputs.json")
    report_commit = _git_timestamp("reports/latest.md")
    existing_date = _existing_report_date()

    should_run = True
    reason = "čerstvé vstupy jsou připravené"

    if event == "schedule" and not _near_local_checkpoint(now):
        should_run = False
        reason = "tento UTC cron je jen DST pojistka; nejde o lokální checkpoint"
    elif event == "schedule" and (trigger is None or trigger.date().isoformat() != today):
        should_run = False
        reason = "dnešní ceny/IV ještě nebyly nahrány"
    elif event == "workflow_dispatch" and not force and (
        trigger is None or trigger.date().isoformat() != today
    ):
        should_run = False
        reason = "ruční běh bez volby force a bez dnešního _trigger"
    elif event == "schedule" and input_commit is None:
        should_run = False
        reason = "nelze ověřit commit s denními vstupy"
    elif (
        not force
        and existing_date == today
        and input_commit is not None
        and report_commit is not None
        and report_commit >= input_commit
    ):
        should_run = False
        reason = "dnešní vstupy už mají vygenerovaný report"

    _set_output("run_report", "true" if should_run else "false")
    _set_output("report_date", today)
    _set_output("reason", reason)
    print(
        f"event={event} local_now={now.isoformat()} "
        f"input_commit={input_commit.isoformat() if input_commit else 'missing'} "
        f"report_commit={report_commit.isoformat() if report_commit else 'missing'} "
        f"existing_report_date={existing_date or 'missing'} "
        f"decision={'run' if should_run else 'skip'} reason={reason}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
