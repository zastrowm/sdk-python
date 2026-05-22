#!/usr/bin/env bash
set -euo pipefail

# Create namespaced python/v* tags pointing to the same commits as bare v* tags.
# This allows the monorepo to distinguish Python releases from other packages.
#
# Usage:
#   cd sdk-python
#   bash create-namespaced-tags.sh
#   git push origin --tags

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

created=0
skipped=0

for tag in $(git tag -l 'v*'); do
  commit=$(git rev-list -n1 "$tag")
  if git rev-parse "python/$tag" >/dev/null 2>&1; then
    skipped=$((skipped + 1))
  else
    git tag "python/$tag" "$commit"
    created=$((created + 1))
  fi
done

echo "Created $created namespaced python/v* tags ($skipped already existed)"
echo ""
echo "To push: git push origin --tags"
