# gemini-code-review-action
A container GitHub Action to review a pull request by Gemini AI.

If the size of a pull request is over the maximum chunk size of the Gemini API, the Action will split the pull request into multiple chunks and generate review comments for each chunk.
And then the Action summarizes the review comments and posts a review comment to the pull request.

## Pre-requisites
We have to set a GitHub Actions secret `GEMINI_API_KEY` to use the Gemini API so that we securely pass it to the Action.

## Inputs

- `gemini_api_key`: The Gemini API key to access the Gemini API [(GET MY API KEY)](https://makersuite.google.com/app/apikey).
- `github_token`: The GitHub token to access the GitHub API (You do not need to generate this Token!).
- `github_repository`: The GitHub repository to post a review comment.
- `github_pull_request_number`: The GitHub pull request number to post a review comment.
- `git_commit_hash`: The git commit hash to post a review comment.
- `pull_request_diff_file`: The path to the git diff of the pull request to generate a review comment.
- `pull_request_diff_chunk_size`: The chunk size of the diff of the pull request to generate a review comment.
- `extra_prompt`: The extra prompt to generate a review comment.
- `model`: The model to generate a review comment. We can use a model which is available.
- `log_level`: The log level to print logs.

As you might know, a model of Gemini has limitation of the maximum number of input tokens.
So we have to split the diff of a pull request into multiple chunks, if the size of the diff is over the limitation.
We can tune the chunk size based on the model we use.

## Example usage
Here is an example to use the Action to review a pull request of the repository.
The actual file is located at [`.github/workflows/ai-code-review.yml`](.github/workflows/ai-code-review.yml).
We set `extra_prompt` to `Sempre responda em português brasileiro!`.
We aim to make GPT review a pull request from a point of view of a Python developer.

As a result of an execution of the Action, the Action posts a review comment to the pull request like the following image.
![An example comment of the code review](./docs/images/example.png)

```yaml
name: "Code Review by Gemini AI"

on:
  pull_request:
env:
  GEMINI_MODEL: "gemini-1.5-pro-latest"

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    env:
      PR_DIFF_PATH: "pull-request.diff"
    steps:
      - uses: actions/checkout@v4
      - name: "Get diff of the pull request"
        id: get_diff
        shell: bash
        env:
          PULL_REQUEST_HEAD_REF: "${{ github.event.pull_request.head.ref }}"
          PULL_REQUEST_BASE_REF: "${{ github.event.pull_request.base.ref }}"
        run: |-
          git fetch origin "${{ env.PULL_REQUEST_HEAD_REF }}"
          git fetch origin "${{ env.PULL_REQUEST_BASE_REF }}"
          git checkout "${{ env.PULL_REQUEST_HEAD_REF }}"
          git diff "origin/${{ env.PULL_REQUEST_BASE_REF }}" > "${{ env.PR_DIFF_PATH }}"
      - uses: Stone-IT-Cloud/gemini-code-review-action@1.0.3
        name: "Code Review by Gemini AI"
        id: review
        with:
          gemini_api_key: ${{ secrets.GEMINI_API_KEY }}
          github_token: ${{ secrets.GITHUB_TOKEN }}
          github_repository: ${{ github.repository }}
          github_pull_request_number: ${{ github.event.pull_request.number }}
          git_commit_hash: ${{ github.event.pull_request.head.sha }}
          model: "gemini-1.5-pro-latest"
          pull_request_diff: |-
            ${{ steps.get_diff.outputs.pull_request_diff }}
          pull_request_chunk_size: "100000"
          extra_prompt: |-
            Always response in English
          log_level: "DEBUG"
```
