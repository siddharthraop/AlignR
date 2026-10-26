# ADR 0005: Gate statuses and run records on the Git path

- Status: Accepted · Date: 2026-10-26

## Context
Pull requests opened with the workflow's `GITHUB_TOKEN` don't trigger other workflows, so a `pull_request` check
workflow wouldn't run on AlignR's own pull requests.

## Decision
The `alignr` workflow runs every gate itself and posts five commit statuses on each pull request. The `gates` workflow
posts the same contexts again when a writer pushes. The run record, with the time from the issue to the last pull
request and the cost, is committed to `runs/`; merges append to `runs/merges.jsonl`.

## Consequences
The scorecard reads plain files in Git, so it needs no database for the Git path. Requiring `alignr/open-questions`
in a branch ruleset makes "0 open questions at merge" a rule, not a hope.
