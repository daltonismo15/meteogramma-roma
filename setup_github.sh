#!/usr/bin/env bash
# Puts this folder on GitHub as a new private repository and starts the first meteogram.
#   usage:  bash setup_github.sh [name]        (default name: meteogramma-roma)
# Needs git. With the GitHub CLI (gh, logged in) it does everything by itself; without it, it explains the 2 steps left.
set -euo pipefail
NAME="${1:-meteogramma-roma}"
cd "$(dirname "$0")"

if [ ! -d .git ]; then
  git init -q
  git checkout -q -b main
fi
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-meteogram}" -c user.email="${GIT_AUTHOR_EMAIL:-meteogram@users.noreply.github.com}" \
    commit -q -m "Rome multi-model meteogram" || true

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  # the token must be allowed to push workflow files
  gh auth status 2>&1 | grep -q "workflow" || gh auth refresh -h github.com -s workflow
  gh repo create "$NAME" --private --source . --remote origin --push
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
  # the workflow saves the meteogram in the repository: it needs write permission
  gh api -X PUT "repos/$REPO/actions/permissions/workflow" -f default_workflow_permissions=write >/dev/null
  echo "starting the first meteogram ..."
  for i in 1 2 3 4 5 6; do
    if gh workflow run meteogram.yml -R "$REPO" >/dev/null 2>&1; then started=1; break; fi
    sleep 5
  done
  echo
  echo "done: https://github.com/$REPO"
  if [ "${started:-0}" = 1 ]; then
    echo "the first meteogram is being made: in 3-5 minutes it is on that page (follow it with: gh run watch -R $REPO)"
  else
    echo "start the first one from the page: Actions -> Rome meteogram -> Run workflow"
  fi
  echo "then every day at 09:30 (10:30 with summer time)"
else
  cat <<MSG

The GitHub CLI (gh) is not installed or not logged in. Two steps left:

 1. On github.com create a new EMPTY repository called $NAME (private is fine;
    no README, no .gitignore), then here run:

      git remote add origin https://github.com/YOUR-USER/$NAME.git
      git push -u origin main

    (the password asked is a personal access token with the "repo" and "workflow" permissions:
     github.com -> Settings -> Developer settings -> Personal access tokens)

 2. On the repository page: Settings -> Actions -> General -> Workflow permissions ->
    "Read and write permissions" -> Save. Then Actions -> Rome meteogram -> Run workflow.

Or install gh (macOS: brew install gh), run "gh auth login", and run this script again.
MSG
fi
