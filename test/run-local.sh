#!/bin/bash

# Build Docker image
# docker build -t gemini-code-review .

# Read test diff
DIFF_CONTENT=$(cat test/long-diff.txt)

mkdir -p test/fixtures
echo "$DIFF_CONTENT" >test/fixtures/test.diff

# Create temp dir for outputs
mkdir -p test/outputs

# Run action locally
docker run \
    -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
    -e GITHUB_TOKEN="your-github-token" \
    -e GITHUB_REPOSITORY="owner/repo" \
    -e GITHUB_PULL_REQUEST_NUMBER="1" \
    -e GIT_COMMIT_HASH="abc123" \
    -v $(pwd)/test/outputs:/outputs \
    -v $(pwd)/test/fixtures/test.diff:/tmp/pr.diff \
    gemini-code-review \
    --model="gemini-1.5-pro-latest" \
    --extra-prompt="Please write your review in English as an experienced nodejs and typescript developer." \
    --temperature=0.7 \
    --max-tokens=250 \
    --top-p=1 \
    --frequency-penalty=0.0 \
    --presence-penalty=0.0 \
    --diff-chunk-size=2000000
