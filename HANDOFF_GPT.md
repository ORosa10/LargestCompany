# Handoff pro ChatGPT — LargestCompany

> **▶ START ZDE (GPT):** Denní report generuje **GitHub Action v tomto repu**
> (běží, jen když nahraješ ten den ceny) a ukládá ho do **`reports/latest.md`**.
> To je nezávislé na jakémkoli AI. ChatGPT jen ten hotový soubor přečte a ukáže.
> (Původní Claude verze je v `HANDOFF_SCHEDULED_TASK.md`, už jen historická.)

Last updated: 2026-09-22

## Kde report je
Report už teď končí v repu — netřeba ho nikam „doručovat":
`https://raw.githubusercontent.com/ORosa10/LargestCompany/main/reports/latest.md`

## ChatGPT Task (volitelné doručení do chatu)
V ChatGPT appce založ **Task**, rozvrh **jednou denně odpoledne** (např. 17:00),
s tímto promptem:

```
Jsi naplánovaná úloha v ChatGPT. Jednou denně:
1. Otevři (browsing) soubor:
   https://raw.githubusercontent.com/ORosa10/LargestCompany/main/reports/latest.md
2. Najdi v něm datum reportu (formát YYYY-MM-DD).
3. POKUD je datum == dnešek: ukaž mi CELÝ obsah souboru tak, jak je (jsou to
   Markdown tabulky — zachovej je), pod nadpisem "LargestCompany — denní report".
   Na začátek přidej jednu českou větu s verdiktem (FAVORABLE / MARGINAL /
   UNFAVORABLE) a obchodovaným tickerem + stranou.
4. POKUD datum != dnešek: napiš jen jednu větu, že dnešní report ještě není
   (ceny nebyly nahrané) — nikdy neukazuj starý report jako dnešní.
5. Nikdy nevymýšlej čísla ani nespouštěj vlastní simulaci.
```

> **Pozn.:** Původní Claude úloha běžela 3× denně a hlídala si, aby report
> poslala „právě jednou" (podle git commit času). ChatGPT Tasks nemají spolehlivou
> paměť mezi běhy, takže tady je to zjednodušené na **1 běh denně** — vyhne se to
> duplicitám. Pokud chceš víc kontrol denně bez rizika duplicit, patří to spíš do
> GitHub Action (mail/commit), ne do ChatGPT.

## Jak něco změnit
- Generování reportu je beze změny (GitHub Action v tomto repu, spouští se při
  nahrání cen). ChatGPT tu jen čte hotový `reports/latest.md`.
