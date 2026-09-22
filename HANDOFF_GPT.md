# Handoff pro ChatGPT — LargestCompany

> **▶ START ZDE (GPT):** Výpočet a report nyní běží v GitHub Actions. ChatGPT
> pouze přečte hotový soubor reports/latest.md a zobrazí ho. Původní cloudová
> varianta je zachována už jen historicky v HANDOFF_SCHEDULED_TASK.md.

Last updated: 2026-09-22

## Jak to teď funguje

1. Ondřej vloží aktuální ceny/IV a Polymarket ceny do daily_inputs.json.
2. .github/workflows/daily.yml se spustí okamžitě po změně tohoto souboru.
3. Stejná Action má plánované kontrolní běhy v lokálním čase 09:30, 12:30
   a 17:00 Europe/Prague. Kvůli letnímu/zimnímu času používá několik UTC
   kandidátů; scripts/report_freshness.py přijme jen správný lokální termín.
4. Pokud jsou vstupy z dneška nové, daily_report.py provede celý výpočet
   a uloží:
   - reports/YYYY-MM-DD.md,
   - reports/latest.md,
   - podpůrný stav do saved_state/ a markets/.
5. Action změny commitne do main a zároveň nahraje report jako GitHub Actions
   artifact s třicetidenní retencí. Staré vstupy se nikdy automaticky
   nepřepočítávají jako nový dnešní report.

Report je veřejně dostupný tady:

https://raw.githubusercontent.com/ORosa10/LargestCompany/main/reports/latest.md

Akce a její běhy jsou tady:

https://github.com/ORosa10/LargestCompany/actions

## ChatGPT Task (volitelné doručení do chatu)

V ChatGPT appce lze založit jeden denní Task, například na 17:00
Europe/Prague, s tímto promptem:

~~~
Jednou denně otevři (browsing) soubor
https://raw.githubusercontent.com/ORosa10/LargestCompany/main/reports/latest.md.
Najdi v něm datum reportu ve formátu YYYY-MM-DD. Pokud je datum stejné jako
dnešek, ukaž mi CELÝ obsah souboru tak, jak je, včetně Markdown tabulek, pod
nadpisem „LargestCompany — denní report“. Na začátek přidej jednu krátkou
českou větu s verdiktem (FAVORABLE / MARGINAL / UNFAVORABLE) a obchodovaným
tickerem + stranou. Pokud datum není dnešek, napiš pouze jednu větu, že dnešní
report ještě není, protože dnešní ceny nebyly nahrané; nikdy nezobrazuj starý
report jako dnešní. Nikdy nevymýšlej čísla ani nespouštěj vlastní simulaci.
Pokud soubor nejde načíst nebo má neočekávaný formát, napiš přesnou chybu
místo domýšlení.
~~~

## Jak něco změnit

- Rozvrh GitHub Action: .github/workflows/daily.yml.
- Kontrola čerstvosti a ochrana před duplikací:
  scripts/report_freshness.py.
- Výpočet modelu, formát reportu a analýza rizika: daily_report.py.
- Vstupy pro další běh: daily_inputs.json.
- Ruční ověření: GitHub → Actions → Daily report → Run workflow.
  Volbu force používej jen pro debugging; běžný běh má vyžadovat dnešní
  _trigger.
