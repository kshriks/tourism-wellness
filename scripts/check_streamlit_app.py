"""
check_streamlit_app.py
------------------------
Streamlit Community Cloud has no public "deploy" API — it isn't a separate
push target. Once an app is connected to this GitHub repo (one-time setup
at share.streamlit.io), it redeploys automatically whenever `main` is
updated, which is exactly what deploy_github.sh's `git push` does. There is
nothing extra to push.

What this script does instead, after you've pushed with deploy_github.sh:
  1. Polls your app's public URL until it responds (a redeploy can take a
     couple of minutes), so you get a clear "it's live" confirmation
     instead of guessing.
  2. Handles app hibernation: Community Cloud puts an app to sleep after
     12 hours with no traffic, and sleeping apps show a wake-up page
     instead of a normal 200 response on the first request. This script
     recognizes that page and keeps polling until the real app loads.

Usage:
    python scripts/check_streamlit_app.py <your-app>.streamlit.app
    python scripts/check_streamlit_app.py <your-app>.streamlit.app --timeout 300
"""

import sys
import time
import argparse
import urllib.request
import urllib.error

# Text Streamlit's hibernation / wake-up page contains, distinguishing a
# sleeping app (still HTTP 200, but not your app) from a live one.
SLEEP_MARKERS = ("Zzzz", "wake", "get this app back up")


def check_once(url: str) -> tuple[bool, str]:
    """Returns (is_live, detail)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(4096).decode("utf-8", errors="ignore")
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            if any(marker.lower() in body.lower() for marker in SLEEP_MARKERS):
                return False, "app is asleep (hibernating) — waiting for it to wake"
            return True, f"HTTP {resp.status}, app content detected"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"not reachable yet ({e.reason})"
    except TimeoutError:
        return False, "request timed out"


def wait_for_app(url: str, timeout: int = 300, interval: int = 10) -> bool:
    if not url.startswith("http"):
        url = f"https://{url}"

    print(f"Checking {url} (timeout {timeout}s, polling every {interval}s)")
    start = time.time()
    while time.time() - start < timeout:
        is_live, detail = check_once(url)
        elapsed = int(time.time() - start)
        if is_live:
            print(f"[{elapsed}s] Live: {detail}")
            print(f"\nApp is up: {url}")
            return True
        print(f"[{elapsed}s] Not ready: {detail}")
        time.sleep(interval)

    print(f"\nTimed out after {timeout}s. Check the app's Cloud logs at "
          f"share.streamlit.io if this persists.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Your Streamlit app's URL, e.g. your-app.streamlit.app")
    parser.add_argument("--timeout", type=int, default=300, help="Max seconds to wait (default 300)")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between checks (default 10)")
    args = parser.parse_args()

    ok = wait_for_app(args.url, args.timeout, args.interval)
    sys.exit(0 if ok else 1)
