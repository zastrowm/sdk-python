# Pull Request Templates

This directory contains multiple PR templates to streamline the contribution process for different types of changes.

## Available Templates

### 1. Default Template (`default.md`)

**Use for:** General documentation changes that don't fall into the other categories.

**Includes:**
- Description & Context section
- Type of Change checkboxes
- Essential checklist items
- Licensing statement

**Auto-selected when:** No template query parameter is provided.

### 2. Quick Fix Template (`quick-fix.md`)

**Use for:** Small, straightforward fixes like:
- Typo corrections
- Formatting adjustments
- Link fixes
- Minor wording improvements

**Includes:**
- Brief description
- Type of Fix checkboxes
- Files Changed section
- Minimal checklist (2 items)
- Licensing statement

**To use:** Add `?template=quick-fix.md` to the PR creation URL.

### 3. Feature Template (`feature.md`)

**Use for:** Major documentation additions like:
- New pages or sections
- Comprehensive guides
- Architectural documentation
- Significant content restructuring

**Includes:**
- Description & Motivation section
- Related Issues linking
- Type of Change checkboxes
- Content Overview section
- Comprehensive Testing & Validation checklist
- Licensing statement

**To use:** Add `?template=feature.md` to the PR creation URL.

## How to Select a Template

When creating a pull request on GitHub:

1. Click the "New Pull Request" button
2. To use a specific template, append the query parameter to the URL:
   - **Quick Fix:** `?template=quick-fix.md`
   - **Feature:** `?template=feature.md`
   - **Default:** No query parameter needed (auto-selected)

**Example URLs:**
```
https://github.com/strands-agents/docs/compare/main...your-branch?template=quick-fix.md
https://github.com/strands-agents/docs/compare/main...your-branch?template=feature.md
```

Alternatively, you can delete the auto-filled template content and manually copy the template you want from this directory.

## For Repository Maintainers

### Deployment Instructions

These templates are currently located in `.github_temp/PULL_REQUEST_TEMPLATE/` and need to be moved to `.github/PULL_REQUEST_TEMPLATE/` to be activated.

**Steps to deploy:**

1. Review the templates in this PR
2. Once approved and merged, create a new PR or commit that:
   - Moves all files from `.github_temp/PULL_REQUEST_TEMPLATE/` to `.github/PULL_REQUEST_TEMPLATE/`
   - Removes the old `.github/PULL_REQUEST_TEMPLATE.md` file (if it exists)
   - Removes the `.github_temp/` directory

3. After deployment, test the templates:
   - Create a test PR without query parameters (should use default template)
   - Create a test PR with `?template=quick-fix.md` (should use quick-fix template)
   - Create a test PR with `?template=feature.md` (should use feature template)

### Why `.github_temp/`?

Templates are initially created in `.github_temp/` because direct changes to `.github/` may require additional permissions or review processes. This approach allows:
- Review of templates before activation
- Testing of markdown rendering
- Validation of content and structure
- Separation of template design from deployment

## Design Rationale

These templates were designed based on analysis of recent merged PRs, which revealed:
- Many template sections were frequently left empty (Areas Affected, Screenshots, Additional Notes)
- Placeholder text was often not removed
- Checklist items were often left unchecked
- A single template was too rigid for varied PR types (typos vs major features)

**Key improvements:**
- ✅ Simplified checklists (3-4 items vs 6+ items)
- ✅ Checkbox format for "Type of Change" (easier than editing text lists)
- ✅ "If applicable" language to reduce checklist pressure
- ✅ Specialized templates for different PR sizes
- ✅ Merged related sections (Description & Context)
- ✅ Removed rarely-used sections

## Feedback

If you have suggestions for improving these templates, please:
- Open an issue describing the problem
- Propose specific changes
- Share examples of PRs where the template could be better

The templates should evolve based on actual contributor usage patterns.
