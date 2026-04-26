# Architectural Decision Records

This directory contains ADRs (Architectural Decision Records) for Atenia Engine. ADRs capture significant architectural decisions, including the context that motivated them, the decision itself, and the consequences expected.

## Why ADRs?

Software projects accumulate decisions. Memory of *why* a decision was made fades faster than memory of *what* was decided. ADRs preserve the reasoning, so future contributors (including the original authors months later) can understand and re-evaluate decisions appropriately.

## Format

ADRs in this project follow the Michael Nygard format:

- **Title** — short, descriptive
- **Status** — Proposed / Accepted / Deferred / Superseded / Rejected
- **Date** — when the decision was made
- **Context** — what is the issue we're addressing?
- **Decision** — what is the position we're taking?
- **Consequences** — what becomes easier or harder as a result?

Additional sections used when relevant: Rationale, Alternatives Considered, Trigger to Revisit, Captured Ideas (for deferred decisions).

## Status meanings

- **Proposed** — under discussion
- **Accepted** — current direction
- **Deferred** — decision postponed pending more information; trigger conditions documented
- **Superseded** — replaced by a later ADR (linked)
- **Rejected** — explicitly decided against

## Index

- [ADR-001](./ADR-001-numerical-health-monitor-deferred.md) — Numerical Health Monitor — Deferred Pending Empirical Data
- [ADR-002](./ADR-002-mathematical-ground-truth-validation.md) — Mathematical Ground-Truth Validation Strategy

## Naming convention

`ADR-NNN-brief-description.md` where `NNN` is zero-padded sequential.
