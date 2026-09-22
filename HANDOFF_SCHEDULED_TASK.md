# Naplánovaná úloha + migrace na nový Claude — handoff

Popisuje Cowork úlohu, která denní „LargestCompany" report **přeposílá** do
chatu, a jak ji obnovit na **novém Claude účtu**. Samotný report generuje
**GitHub Action v tomto repu** (běží, jen když nahraješ ten den ceny) — to
funguje nezávisle na Claude. Úloha níže report jen relayuje; nic nepočítá.

Last updated: 2026-09-22

## Úloha `largestcompany-daily-report`
- **Rozvrh (cron):** `30 9,12,17 * * *` (kontrola v 9:30, 12:30 a 17:00; postne max 1× denně)
- **Závislosti:** tento veřejný repo (klon bez tokenu), soubor `reports/latest.md`.
- **Obnovení na novém Claude:** řekni „Založ tuhle naplánovanou úlohu podle
  dokumentu" — Claude zavolá `create_scheduled_task` s cronem a promptem níže.

## Prompt (doslovně)

```
You relay the daily "LargestCompany" trading report for Ondřej into this chat. Post it exactly ONCE per day, as soon as a fresh report for TODAY exists — whether that day's Polymarket prices were uploaded before 9:30, or later (e.g. around noon). The report is produced by a GitHub Action that runs only when Ondřej uploads that day's prices. Do NOT run any simulation or fetch market data yourself.

You run three times a day: 09:30, 12:30 and 17:00 (local). To post exactly once, each run posts the report ONLY if it was generated (git-committed) SINCE THE PREVIOUS scheduled run.

IMPORTANT: always clone into a FRESH unique directory (never reuse /tmp/lc — a stale or permission-locked leftover would make you read yesterday's report).

Steps (use the workspace bash tool, one script):
1. Clone the public repo into a fresh temp dir (no token needed):
   d=$(mktemp -d) && git clone --depth 1 -q https://github.com/ORosa10/LargestCompany.git "$d" && echo "cloned into $d"
2. Compute today, the report's commit time, and the previous scheduled-run boundary:
   now=$(date +%s); today=$(date +%F)
   committed=$(cd "$d" && git log -1 --format=%ct -- reports/latest.md)
   c1=$(date -d "$today 09:30" +%s); c2=$(date -d "$today 12:30" +%s); c3=$(date -d "$today 17:00" +%s); cy=$(date -d "yesterday 17:00" +%s)
   prev=$cy; for c in $c1 $c2 $c3; do if [ "$c" -le $((now-3600)) ] && [ "$c" -gt "$prev" ]; then prev=$c; fi; done
   report_date=$(grep -m1 -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' "$d/reports/latest.md")
   Also print report_date, committed, prev so the decision is auditable.
3. POST the report ONLY IF: report_date == today AND committed > prev
   (today's report was generated after the previous check → this is the first check that sees it).
   - Post the FULL contents of "$d/reports/latest.md" verbatim (it is markdown tables — post as-is so they render), under the title "LargestCompany — daily report".
   - Precede it with ONE short Czech intro sentence stating the verdict (FAVORABLE / MARGINAL / UNFAVORABLE) and the traded ticker + side.
4. OTHERWISE do nothing and post nothing:
   - report_date != today → no fresh prices uploaded yet today (never post yesterday's report).
   - committed <= prev → today's report was already relayed on an earlier check today.
   Never post a stale report and never fabricate numbers.
```
