# Deploying LargestCompany (auto-update on every push)

The app is a standard Streamlit app whose entry point is `app.py`. Once it is
connected to Streamlit Community Cloud, **every push to `main` redeploys the
live app automatically** - no `git pull` and no manual step ever again.

## One-time setup (about 2 minutes)

1. Go to https://share.streamlit.io and sign in with the **same GitHub account**
   that owns `ORosa10/LargestCompany`.
2. Click **Create app** -> **Deploy a public/private app from GitHub**.
3. When prompted, **authorize Streamlit to access your GitHub** (needed once for
   private repos).
4. Fill in:
   - Repository: `ORosa10/LargestCompany`
   - Branch: `main`
   - Main file path: `app.py`
5. (Optional) Under **Advanced settings**, pick Python 3.12.
6. Click **Deploy**. First build installs `requirements.txt` (a few minutes).

You get a URL like `https://largestcompany.streamlit.app`. Bookmark it.

## From then on

- Every `git push origin main` (including everything Claude pushes) triggers an
  automatic redeploy. Open the URL and click **Reload** if the tab was open.
- The sidebar shows all phases, including Phase 7 and Phase 8, with no pull.

## Notes

- The app persists run state under the home directory (`~/.largestcompany`).
  On Streamlit Cloud this is ephemeral and resets on each redeploy, which is
  fine - rerun Phase 1 to repopulate.
- No secrets or API keys are required; market data comes from Yahoo via
  `yfinance` over the network, which Streamlit Cloud allows.
- Requirements are pinned by lower bound in `requirements.txt`.
