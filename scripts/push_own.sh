#!/bin/bash
# Push the files of this run's last commit onto the newest main as one commit touching only those files. Never a merge:
# a merge carries whatever landed upstream meanwhile into what this run pushes, and on 17 Oct that was a workflow file,
# which the job's token may not push, so the build's index.json stayed behind while its index.bin went up.
#   bash scripts/push_own.sh "<commit message>"      (after git commit of the run's files)
set -u
MSG="${1:-this run outputs}"
CH=$(git diff --name-only HEAD~1 HEAD)
[ -n "$CH" ] || { echo "push_own: nothing to push"; exit 0; }
mkdir -p "$RUNNER_TEMP/mine" "$RUNNER_TEMP/dirty"
for f in $CH; do [ -f "$f" ] && mkdir -p "$RUNNER_TEMP/mine/$(dirname "$f")" && cp -a "$f" "$RUNNER_TEMP/mine/$f"; done
# the run's other changes to tracked files, not yet committed (the build rewrites out/index.json and out/detail before the
# log is pushed, and the publish step needs them as the build left them): kept aside and put back after the push
DIRTY=$(git diff --name-only)
for f in $DIRTY; do [ -f "$f" ] && mkdir -p "$RUNNER_TEMP/dirty/$(dirname "$f")" && cp -a "$f" "$RUNNER_TEMP/dirty/$f"; done
restore_dirty() { for f in $DIRTY; do [ -f "$RUNNER_TEMP/dirty/$f" ] && mkdir -p "$(dirname "$f")" && cp -a "$RUNNER_TEMP/dirty/$f" "$f"; done; rm -rf "$RUNNER_TEMP/dirty"; }
OUT=""
for i in 1 2 3 4 5; do
  git fetch -q origin main && git reset -q --hard origin/main || { sleep $((i*10)); continue; }
  for f in $CH; do
    if [ -f "$RUNNER_TEMP/mine/$f" ]; then mkdir -p "$(dirname "$f")"; cp -a "$RUNNER_TEMP/mine/$f" "$f"; else rm -f "$f"; fi
  done
  git add -A -- $CH 2>/dev/null
  git diff --cached --quiet && { echo "push_own: already on main"; restore_dirty; exit 0; }
  git commit -qm "$MSG"
  if OUT=$(git push -q origin HEAD:main 2>&1); then echo "push_own: pushed"; restore_dirty; exit 0; fi
  echo "push attempt $i failed: $OUT"; sleep $((i*10))
done
restore_dirty
echo "::error title=push::could not push this run's files: $(echo "$OUT" | tail -2 | tr '\n' ' ' | cut -c1-300)"; exit 1
