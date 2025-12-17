# Atenia Engine — Research Paper (Preprint)

This directory contains the current public preprint of the Atenia Engine research paper.

## Status

- 📘 Preprint — Publicly available
- ⏳ arXiv submission pending endorsement
- 🧾 Patent pending (USPTO provisional filed)

## Reproducibility

All experimental results presented in the paper are implemented as executable tests in the main repository.

To reproduce results:

```bash
cargo test -- --nocapture

Optional debug mode:

$env:ATENIA_DEBUG="1"
cargo test -- --nocapture

If the tests pass, the execution engine described in the paper is alive.