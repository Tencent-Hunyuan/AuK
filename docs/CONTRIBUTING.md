# Contributing to AuK

Thank you for helping improve AuK. Contributions may include bug reports,
documentation, tests, inference fixes, user-interface improvements, performance
work, and compatibility fixes.

All participation in the AuK community must follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Before starting

1. Search existing issues and pull requests before opening a new one.
2. Open an issue before investing substantial work in any of the following:
   - a new task, model architecture, or checkpoint format;
   - a public API or configuration change;
   - a new runtime dependency;
   - a change that affects model output, memory use, or inference speed; or
   - a large refactor.
3. Bugs, documentation corrections, and narrowly scoped fixes may be submitted
   directly when their purpose and validation are clear.

Early discussion is a scope check: it helps contributors and maintainers agree
on the problem, constraints, and acceptance criteria before implementation.

## Reporting bugs

Use the repository's **Bug report** issue form. A useful report includes:

- the affected area and AuK variant (`AuK` or `AuK-Flash`);
- checkpoint source and exact revision or commit;
- operating system, Python, PyTorch, CUDA, GPU, and driver versions;
- installation method and relevant dependency versions;
- the smallest input and complete command that reproduce the problem;
- expected behavior and actual behavior; and
- the complete traceback or logs, with secrets and private data removed.

For audio-related bugs, describe the format, sample rate, channel count, and
duration. Share only audio that the reporter has the right to disclose. A
minimal synthetic or otherwise non-sensitive sample is preferred.

Do not post access tokens, private model URLs, personal speech, or confidential
data in an issue. Security vulnerabilities and Code of Conduct incidents should
be reported privately to the maintainers.

## Development setup

Create a fork, clone it, and work on a focused branch:

```bash
git clone <your-fork-url>
cd AuK
git switch -c <short-descriptive-branch>
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m pip install ruff pre-commit
pre-commit install
```

Alternatively, use conda, which is often more convenient for installing a
CUDA-enabled PyTorch build:

```bash
conda create -n auk python=3.10
conda activate auk
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m pip install ruff pre-commit
pre-commit install
```

For changes to the fine-tuning pipeline, install the training dependencies:

```bash
python3 -m pip install -e ".[train]"
```

GPU-specific PyTorch installations may require the platform-appropriate wheel.
Record any deviation from the dependencies declared in `pyproject.toml` in the
pull request.

The application downloads model assets at runtime. Follow the environment and
repository-access instructions in [README.md](README.md), and keep credentials
in environment variables or repository secrets. Credentials must never be
committed to the repository.

## Testing and code quality

Run the smallest checks that cover the change, then report the exact commands
and results in the pull request.

Ruff is configured in `pyproject.toml` for Python 3.10 with a line length of
130. The bundled VAE implementation has explicit exclusions and per-file
ignores; do not broaden those exceptions without explaining the reason in the
pull request.

Before requesting review, run the repository quality gate:

```bash
ruff check .
ruff format --check .
pre-commit run --all-files
python3 -m compileall -q src
```

The pre-commit hooks run Ruff lint fixes, Ruff formatting, import sorting, and
YAML validation. Some hooks update files in place. Review those edits, stage
them, and rerun `pre-commit run --all-files` until every hook passes.

Changes to the Gradio interface should also pass this lightweight UI smoke test:

```bash
python3 - <<'PY'
from auk.infer.infer_gradio import build_demo

demo = build_demo()
assert demo is not None
print("Gradio UI smoke test passed.")
PY
```

Changes to model loading, inference, sampling, audio I/O, or checkpoint handling
require at least one representative end-to-end inference. Test every affected
variant. Include the checkpoint revision, GPU, input description, command,
result, and relevant memory or latency observations in the pull request. If the
required hardware or weights are unavailable, state that limitation explicitly.

The repository does not currently include a project-wide unit-test suite. Add a
targeted regression test when a suitable test boundary exists. Otherwise,
include the minimal reproducer and its before-and-after result in the pull
request. Keep diffs focused and avoid unrelated formatting or generated edits.

## Pull requests

Each pull request should:

- solve one coherent problem;
- link the relevant issue when one exists;
- explain the motivation and user-visible behavior;
- describe the implementation at the level needed for review;
- list all validation commands and results;
- add or update tests when practical;
- update documentation when behavior, configuration, or interfaces change;
- identify compatibility, quality, latency, and memory effects; and
- avoid generated files, model weights, datasets, caches, and unrelated edits.

Draft pull requests are welcome for early technical feedback. A pull request is
ready for review when its description and validation allow another contributor
to understand and reproduce the change.

## Model, data, and audio contributions

Discuss new model implementations, training recipes, checkpoints, datasets, and
task formats in an issue before submitting code.

- Do not commit checkpoints, datasets, generated audio collections, or other
  large artifacts unless maintainers have explicitly approved the location and
  distribution method.
- Document the source, version, license, and intended use of third-party code,
  models, and data.
- Confirm that submitted audio and data may legally and ethically be shared for
  the proposed use.
- Preserve provenance. Evaluation results should state the model revision,
  dataset or input set, metric implementation, and execution settings.
- Keep secrets, private URLs, user data, and identifying metadata out of commits.

## AI-assisted contributions

AI tools may assist development, but the human contributor remains responsible
for every submitted line and claim. Contributors using such tools should:

- reproduce and understand the problem before changing code;
- review the complete diff and remove unrelated generated changes;
- understand and be able to explain the implementation;
- verify licenses and provenance for suggested code or text;
- run the relevant checks themselves; and
- disclose material AI assistance when it helps reviewers assess provenance or
  validation.

Low-value mechanical changes, fabricated test results, unexplained generated
code, and broad edits without a reproducible purpose may be closed.

## Review expectations

Maintainers may request a smaller scope, additional reproduction evidence,
tests, documentation, or compatibility checks. Review discussion should stay
focused on the code and its effects. Acceptance is based on project fit,
correctness, maintainability, evidence, and available maintainer capacity.
