#!/usr/bin/env bash
# deploy_github.sh
# -----------------
# Pushes this project folder to your existing GitHub repository. Run it
# from inside the tourism_mlops/ folder. Safe to re-run: it only inits git
# or adds the "origin" remote if they aren't already set up, and it skips
# committing if there's nothing new.
#
# Usage:
#   ./scripts/deploy_github.sh <your-existing-repo-url>
#
# Example:
#   ./scripts/deploy_github.sh git@github.com:mahesh/wellness-tourism-mlops.git
#
# Pushing to main triggers .github/workflows/pipeline.yml, which validates
# the data, retrains the model, and commits it back to main. Streamlit
# Community Cloud, once connected to this repo, redeploys automatically on
# every push to main — see check_streamlit_app.py to confirm when that
# redeploy is actually live.

set -e

REPO_URL="$1"
if [ -z "$REPO_URL" ]; then
  echo "Usage: $0 <your-existing-repo-url>"
  exit 1
fi

if [ ! -d .git ]; then
  git init
  git checkout -b main
fi

if ! git remote | grep -q '^origin$'; then
  git remote add origin "$REPO_URL"
fi

git add .
git commit -m "Deploy Wellness Tourism Package MLOps pipeline" || echo "Nothing new to commit"
git push -u origin main

echo ""
echo "Pushed to $REPO_URL"
echo "Check the Actions tab on GitHub — pipeline.yml should now be running."
