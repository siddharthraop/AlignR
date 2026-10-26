# ADR 0001: Reuse the KnowledgeFlow runtime, not ADK

- Status: Accepted · Date: 2026-10-26 · Deciders: Siddharth Rao

## Context
The design names Agent Development Kit as the framework. AlignR's loop is fixed: one model tool and nine deterministic
tools in a known order, with at most two revisions. KnowledgeFlow shipped the same shape with `google-genai` directly.

## Decision
The loop is plain Python in `alignr.agent`, and the model is called with `google-genai`. Each tool in the design is a
function with the same name. The runtime is KnowledgeFlow's: Cloud Run, 0 to 1 instances, the shared runtime service
account and the `gemini-api-key` secret.

## Consequences
A smaller image and a loop that tests can drive step by step. Moving to ADK later means wrapping the same functions as
ADK tools; nothing in the checks changes.
