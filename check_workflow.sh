#!/usr/bin/env bash
set -euo pipefail

# Get the latest workflow run for the current branch
BRANCH=$(git branch --show-current)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

echo "Monitoring latest workflow run on branch '${BRANCH}' in ${REPO}..."

# Get latest workflow run ID
RUN_ID=$(gh run list --branch "$BRANCH" --limit 1 --json databaseId -q '.[0].databaseId')

if [ -z "$RUN_ID" ]; then
    echo "No workflow runs found for branch ${BRANCH}"
    exit 1
fi

# Poll until workflow completes
while true; do
    STATUS=$(gh run view "$RUN_ID" --json status,conclusion -q '.status')
    CONCLUSION=$(gh run view "$RUN_ID" --json status,conclusion -q '.conclusion')

    if [ "$STATUS" != "completed" ]; then
        echo "Workflow still running... (status: $STATUS)"
        sleep 5
        continue
    fi

    # Workflow complete - check conclusion
    if [ "$CONCLUSION" = "success" ]; then
        echo "✅ Workflow completed successfully!"
        exit 0
    else
        echo "❌ Workflow failed (conclusion: $CONCLUSION)"
        exit 1
    fi
done
