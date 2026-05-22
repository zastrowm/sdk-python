#!/usr/bin/env bash
set -euo pipefail

# Prepare sdk-python for monorepo convergence.
# This script reproduces the full pre-merge transformation:
#   1. Move all source into strands-py/ subdirectory (keep README at root)
#   2. Rename Python-specific workflows with python- prefix, add path filters + working-directory
#      (generic repo-level workflows like issue management stay unprefixed)
#   3. Configure hatch VCS versioning for monorepo (python/v* tags)
#   4. Update dependabot directory
#
# Tag creation is handled separately by create-namespaced-tags.sh.
#
# Usage:
#   git clone https://github.com/strands-agents/sdk-python.git
#   cd sdk-python
#   bash prepare-monorepo.sh

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -d strands-py ]]; then
  echo "ERROR: strands-py/ already exists. Is this script being run twice?" >&2
  exit 1
fi

echo "=== Step 1: Move all files into strands-py/ ==="

mkdir strands-py

# Move all top-level items except .git, .github, strands-py, and scripts
for item in *; do
  [[ "$item" == "strands-py" ]] && continue
  [[ "$item" == "prepare-monorepo.sh" ]] && continue
  [[ "$item" == "create-namespaced-tags.sh" ]] && continue
  [[ "$item" == "README.md" ]] && continue
  git mv "$item" strands-py/
done
# Move hidden tracked files (excluding .git and .github)
for item in .[!.]*; do
  [[ "$item" == ".git" ]] && continue
  [[ "$item" == ".github" ]] && continue
  # Only move if tracked by git
  if git ls-files --error-unmatch "$item" >/dev/null 2>&1; then
    git mv "$item" strands-py/
  fi
done

# Write root .gitignore (subset — full Python ignores stay in strands-py/)
cat > .gitignore <<'ROOTIGNORE'
.DS_Store
__pycache__*
.env
.venv
*.bak
.vscode
.kiro
CLAUDE.md
.claude/settings.local.json
ROOTIGNORE

git add .gitignore

git commit -m "$(cat <<'EOF'
chore: move source into strands-py/ subdirectory

Relocate all project files (src, tests, docs, config) into the
strands-py/ subdirectory in preparation for monorepo convergence.
Keep README at root. Workflows remain at .github/ root and will be
updated in a subsequent commit.
EOF
)"

echo "=== Step 2: Rename Python-specific workflows and add path filters ==="

# Python-specific workflows get the python- prefix
PYTHON_WORKFLOWS=(
  "pr-and-push.yml"
  "test-lint.yml"
  "pypi-publish-on-release.yml"
  "integration-test.yml"
  "publish-lambda-layer.yml"
  "check-markdown-links.yml"
)

for wf in "${PYTHON_WORKFLOWS[@]}"; do
  if [[ -f ".github/workflows/$wf" ]]; then
    git mv ".github/workflows/$wf" ".github/workflows/python-$wf"
  fi
done

# Rename the SOP doc too
if [[ -f .github/workflows/LAMDBA_LAYERS_SOP.md ]]; then
  git mv .github/workflows/LAMDBA_LAYERS_SOP.md .github/workflows/python-LAMBDA_LAYERS_SOP.md
fi

# --- python-pr-and-push.yml ---
cat > .github/workflows/python-pr-and-push.yml <<'WORKFLOW'
name: "Python: Pull Request and Push"

on:
  pull_request:
    branches: [ main ]
    types: [opened, synchronize, reopened, ready_for_review]
    paths:
      - 'strands-py/**'
      - '.github/workflows/python-*'
  push:
    branches: [ main ]
    paths:
      - 'strands-py/**'
      - '.github/workflows/python-*'
  workflow_dispatch:
concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  call-test-lint:
    uses: ./.github/workflows/python-test-lint.yml
    permissions:
      contents: read
    with:
      ref: ${{ github.event.pull_request.head.sha || github.sha }}
    secrets:
      CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}

  check-api:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
    - name: Checkout code
      uses: actions/checkout@v6
      with:
        fetch-depth: 0
    - name: Setup uv
      uses: astral-sh/setup-uv@v7
    - name: Check API breaking changes
      run: |
        if ! uvx griffe check --search strands-py/src --format github strands --against "main"; then
          echo "Potential API changes detected (review if actually breaking)"
          exit 1
        fi
WORKFLOW

# --- python-test-lint.yml ---
cat > .github/workflows/python-test-lint.yml <<'WORKFLOW'
name: "Python: Test and Lint"

on:
  workflow_call:
    inputs:
      ref:
        required: true
        type: string
    secrets:
      CODECOV_TOKEN:
        required: false

jobs:
  unit-test:
    name: Unit Tests - Python ${{ matrix.python-version }} - ${{ matrix.os-name }}
    permissions:
      contents: read
    strategy:
      matrix:
       include:
        # Linux
        - os: ubuntu-latest
          os-name: 'linux'
          python-version: "3.10"
        - os: ubuntu-latest
          os-name: 'linux'
          python-version: "3.11"
        - os: ubuntu-latest
          os-name: 'linux'
          python-version: "3.12"
        - os: ubuntu-latest
          os-name: 'linux'
          python-version: "3.13"
        - os: ubuntu-latest
          os-name: 'linux'
          python-version: "3.14"
        # Windows
        - os: windows-latest
          os-name: 'windows'
          python-version: "3.10"
        - os: windows-latest
          os-name: 'windows'
          python-version: "3.11"
        - os: windows-latest
          os-name: 'windows'
          python-version: "3.12"
        - os: windows-latest
          os-name: 'windows'
          python-version: "3.13"
        - os: windows-latest
          os-name: 'windows'
          python-version: "3.14"
        # MacOS - latest only; not enough runners for macOS
        - os: macos-latest
          os-name: 'macOS'
          python-version: "3.14"
      fail-fast: true
    runs-on: ${{ matrix.os }}
    env:
      LOG_LEVEL: DEBUG
    defaults:
      run:
        working-directory: strands-py
    steps:
      - name: Checkout code
        uses: actions/checkout@v6
        with:
          ref: ${{ inputs.ref }}
          persist-credentials: false
      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install system audio dependencies (Linux)
        if: matrix.os-name == 'linux'
        working-directory: .
        run: |
          sudo apt-get update
          sudo apt-get install -y portaudio19-dev libasound2-dev
      - name: Install system audio dependencies (macOS)
        if: matrix.os-name == 'macOS'
        working-directory: .
        run: |
          brew install portaudio
      - name: Install system audio dependencies (Windows)
        if: matrix.os-name == 'windows'
        working-directory: .
        run: |
          # Windows typically has audio libraries available by default
          echo "Windows audio dependencies handled by PyAudio wheels"
      - name: Install dependencies
        run: |
          pip install --no-cache-dir hatch 'virtualenv<21'
      - name: Run Unit tests
        id: tests
        run: hatch test tests --cover
        continue-on-error: false

      - name: Upload coverage reports to Codecov
        uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
  lint:
    name: Lint
    runs-on: ubuntu-latest
    permissions:
      contents: read
    defaults:
      run:
        working-directory: strands-py
    steps:
      - name: Checkout code
        uses: actions/checkout@v6
        with:
          ref: ${{ inputs.ref }}
          persist-credentials: false

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install system audio dependencies (Linux)
        working-directory: .
        run: |
          sudo apt-get update
          sudo apt-get install -y portaudio19-dev libasound2-dev

      - name: Install dependencies
        run: |
          pip install --no-cache-dir hatch 'virtualenv<21'

      - name: Run lint
        id: lint
        run: hatch fmt --linter --check
        continue-on-error: false
WORKFLOW

# --- python-pypi-publish-on-release.yml ---
cat > .github/workflows/python-pypi-publish-on-release.yml <<'WORKFLOW'
name: "Python: Publish Package"

on:
  release:
    types:
      - published

jobs:
  call-test-lint:
    if: startsWith(github.event.release.tag_name, 'python/v')
    uses: ./.github/workflows/python-test-lint.yml
    permissions:
      contents: read
    with:
      ref: ${{ github.event.release.target_commitish }}
    secrets:
      CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}

  build:
    name: Build distribution
    if: startsWith(github.event.release.tag_name, 'python/v')
    permissions:
      contents: read
    needs:
      - call-test-lint
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: strands-py

    steps:
    - uses: actions/checkout@v6
      with:
        persist-credentials: false
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v6
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install hatch twine 'virtualenv<21'

    - name: Validate version
      run: |
        version=$(hatch version)
        if [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "Valid version format"
            exit 0
        else
            echo "Invalid version format"
            exit 1
        fi

    - name: Build
      run: |
        hatch build

    - name: Store the distribution packages
      uses: actions/upload-artifact@v7
      with:
        name: python-package-distributions
        path: strands-py/dist/

  deploy:
    name: Upload release to PyPI
    needs:
    - build
    runs-on: ubuntu-latest

    environment:
      name: pypi
      url: https://pypi.org/p/strands-agents
    permissions:
      id-token: write

    steps:
    - name: Download all the dists
      uses: actions/download-artifact@v8
      with:
        name: python-package-distributions
        path: dist/
    - name: Publish distribution to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
WORKFLOW

# --- python-integration-test.yml ---
cat > .github/workflows/python-integration-test.yml <<'WORKFLOW'
name: "Python: Integration Test"

on:
  pull_request_target:
    branches: [main]
    paths:
      - 'strands-py/**'
      - '.github/workflows/python-*'
  merge_group:
    types: [checks_requested]

jobs:
  authorization-check:
    name: Check access
    permissions: read-all
    runs-on: ubuntu-latest
    outputs:
      approval-env: ${{ steps.auth.outputs.approval-env }}
    steps:
      - name: Check Authorization
        id: auth
        uses: strands-agents/devtools/authorization-check@main
        with:
          skip-check: ${{ github.event_name == 'merge_group' }}
          username: ${{ github.event.pull_request.user.login || 'invalid' }}
          allowed-roles: 'maintain,triage,write,admin'

  check-access-and-checkout:
    runs-on: ubuntu-latest
    needs: authorization-check
    environment: ${{ needs.authorization-check.outputs.approval-env }}
    permissions:
      id-token: write
      pull-requests: read
      contents: read
    defaults:
      run:
        working-directory: strands-py
    steps:
      - name: Configure Credentials
        uses: aws-actions/configure-aws-credentials@v6
        with:
         role-to-assume: ${{ secrets.STRANDS_INTEG_TEST_ROLE }}
         aws-region: us-east-1
         mask-aws-account-id: true

      - name: Checkout head commit
        uses: actions/checkout@v6
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          persist-credentials: false
      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install --no-cache-dir hatch 'virtualenv<21'
      - name: Run integration tests
        env:
          AWS_REGION: us-east-1
          AWS_REGION_NAME: us-east-1
          STRANDS_TEST_API_KEYS_SECRET_NAME: ${{ secrets.STRANDS_TEST_API_KEYS_SECRET_NAME }}
        id: tests
        run: |
          hatch test tests_integ

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v7
        with:
          name: test-results
          path: strands-py/build/test-results.xml

  upload-metrics:
    runs-on: ubuntu-latest
    needs: check-access-and-checkout
    if: always()
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Configure Credentials
        uses: aws-actions/configure-aws-credentials@v6
        with:
         role-to-assume: ${{ secrets.STRANDS_INTEG_TEST_ROLE }}
         aws-region: us-east-1
         mask-aws-account-id: true

      - name: Checkout main
        uses: actions/checkout@v6
        with:
          ref: main
          sparse-checkout: |
            .github/scripts
          persist-credentials: false

      - name: Download test results
        uses: actions/download-artifact@v8
        with:
          name: test-results

      - name: Publish test metrics to CloudWatch
        run: |
          pip install --no-cache-dir boto3
          python .github/scripts/upload-integ-test-metrics.py test-results.xml ${{ github.event.repository.name }}
WORKFLOW

# Update the markdown links checker (Python-specific, prefixed)
python3 << 'PYEOF'
text = open('.github/workflows/python-check-markdown-links.yml').read()
text = text.replace('name: Check Markdown Links', 'name: "Python: Check Markdown Links"')
text = text.replace("config-file: '.markdown-link-check.json'",
     "config-file: 'strands-py/.markdown-link-check.json'\n          folder-path: 'strands-py'")
open('.github/workflows/python-check-markdown-links.yml', 'w').write(text)
PYEOF

# Update the lambda layer workflow name (Python-specific, prefixed)
python3 << 'PYEOF'
text = open('.github/workflows/python-publish-lambda-layer.yml').read()
text = text.replace('name: Publish PyPI Package to Lambda Layer', 'name: "Python: Publish Lambda Layer"')
open('.github/workflows/python-publish-lambda-layer.yml', 'w').write(text)
PYEOF

git add -A
git commit -m "$(cat <<'EOF'
chore: rename Python-specific workflows with python- prefix and add path filters

- Rename Python build/test/publish workflows with python- prefix
- Add path filters (strands-py/**, .github/workflows/python-*) to
  PR/push-triggered Python workflows
- Add working-directory: strands-py for build/test/lint steps
- Update griffe check to use strands-py/src search path
- Add workflow_dispatch trigger to pr-and-push for manual testing
- Scope publish workflow to python/v* tags
- Generic repo-level workflows (issue management, PR labeling, strands
  commands) remain unprefixed and unscoped
EOF
)"

echo "=== Step 3: Configure hatch VCS versioning for monorepo ==="

python3 -c "
text = open('strands-py/pyproject.toml').read()
old = '[tool.hatch.version]\nsource = \"vcs\"  # Use git tags for versioning'
new = '''[tool.hatch.version]
source = \"vcs\"
raw-options.root = \"..\"
raw-options.tag_regex = \"^python/v(?P<version>.+)\$\"
raw-options.git_describe_command = [\"git\", \"describe\", \"--dirty\", \"--tags\", \"--long\", \"--match\", \"python/v*\"]'''
text = text.replace(old, new)
open('strands-py/pyproject.toml', 'w').write(text)
"

git add strands-py/pyproject.toml
git commit -m "$(cat <<'EOF'
chore: configure hatch VCS versioning for monorepo

Update pyproject.toml version settings to work in the monorepo layout:
- root=".." to look at git root one level up from strands-py/
- tag_regex to only match python/v* tags
- git_describe_command constrained to python/v* tags

This is backward-incompatible with the pre-monorepo layout and must
land as part of the restructure.
EOF
)"

echo "=== Step 4: Update dependabot ==="

# Replace the pip ecosystem directory and prefix (first occurrence only)
python3 -c "
text = open('.github/dependabot.yml').read()
text = text.replace('directory: \"/\"', 'directory: \"/strands-py\"', 1)
text = text.replace('prefix: ci', 'prefix: \"ci(python)\"', 1)
open('.github/dependabot.yml', 'w').write(text)
"

git add .github/dependabot.yml
git commit -m "$(cat <<'EOF'
chore: update dependabot pip directory to /strands-py

Point the pip ecosystem at the new subdirectory location.
The github-actions ecosystem stays at / since workflows remain at root.
EOF
)"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Run ./create-namespaced-tags.sh to create python/v* tags"
echo "  2. Push to your fork:  git push origin main --tags"
echo "  3. Trigger workflow:   gh workflow run python-pr-and-push.yml --ref main"
echo "  4. Open a test PR touching strands-py/ to validate path triggers"
