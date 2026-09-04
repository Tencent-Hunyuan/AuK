## Summary and motivation

<!-- Explain the problem, the main changes, and the user-visible outcome. -->

## Related issue

<!-- Use "Closes #123" when merging this PR should close an issue. -->

Closes #

## Change type

- [ ] Bug fix
- [ ] New feature or task
- [ ] Performance or memory improvement
- [ ] Model, checkpoint, or training change
- [ ] Compatibility or dependency change
- [ ] Documentation, refactor, or maintenance

## Validation

<!-- Run every applicable check. Explain any unchecked item below. -->

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `pre-commit run --all-files`
- [ ] `python3 -m compileall -q src`
- [ ] Relevant Gradio, inference, or training test

Commands and results:

```text

```

Checks not run and reason: <!-- Write "None" when all applicable checks ran. -->

## Model and runtime validation

<!-- Complete for behavior-affecting changes; write N/A otherwise. -->

- AuK variant: <!-- Base / Flash / Both / N/A -->
- Checkpoint source and revision:
- GPU and VRAM:
- Input or audio description:
- Inference or training command:
- Observed result:

## Impact

<!-- Describe measurable effects or write N/A. -->

- Output quality:
- Latency and GPU memory:
- Public API or configuration:
- Checkpoint or data compatibility:
- New dependencies:

## Documentation and provenance

- [ ] User-facing behavior, configuration, examples, and comments are updated where needed.
- [ ] No credentials, private URLs, personal data, model weights, datasets, or generated artifact collections are committed.
- [ ] Third-party code, models, data, and audio have documented sources and compatible licenses.
- [ ] Submitted audio and data may legally and ethically be shared for the proposed use.
- [ ] Material AI assistance is disclosed below, or no material AI assistance was used.

Documentation, provenance, and AI-assistance notes:

## Final checklist

- [ ] The change is focused and the complete diff has been self-reviewed.
- [ ] Unrelated formatting and generated edits have been removed.
- [ ] Another contributor can reproduce the reported validation.
- [ ] The contribution follows `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

## Reviewer notes

<!-- Highlight uncertain areas, trade-offs, or files needing special attention. -->
