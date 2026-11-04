# Releasing sophon-bench

Publishing to PyPI is automated by `.github/workflows/python-publish.yml`. It
runs **only when a GitHub Release is published**: never on a merge to `main`,
never for draft releases, never for prereleases.

- PyPI project: [`sophon-bench`](https://pypi.org/p/sophon-bench)
- Import package: `sophon`
- Version source of truth: `version` in `pyproject.toml`. `sophon.__version__`
  reads it back from the installed package metadata.

Authentication uses **PyPI Trusted Publishing** (OIDC). There is no
`PYPI_API_TOKEN` secret, and there should never be one.

## One-time setup

Both steps are required before the first automated release. Until they are
done, the publish job will fail with a 403 or hang waiting on a missing
environment.

### 1. Register the trusted publisher on PyPI

On <https://pypi.org/manage/project/sophon-bench/settings/publishing/>, add a
GitHub publisher with **exactly** these values:

| Field             | Value                |
| ----------------- | -------------------- |
| Owner             | this repo's GitHub owner |
| Repository        | this repo's name     |
| Workflow name     | `python-publish.yml` |
| Environment name  | `pypi`               |

The workflow name is the filename only, not the full path. The environment
name must match the workflow's `environment.name` exactly or PyPI will reject
the OIDC token.

### 2. Create the `pypi` environment

In **Settings → Environments → New environment**, name it `pypi`, then:

- Enable **Required reviewers** and add yourself, so the upload pauses for
  approval. (On a private repo this needs a paid GitHub plan; without it the
  environment still works for OIDC, just without the approval gate.)
- Optionally restrict **Deployment branches and tags** to tags matching `v*`.

## Cutting a release

1. Bump `version` in `pyproject.toml` and merge to `main`.
2. `make tag`: checks the tree is clean, the tag is free, and the version is
   not on PyPI, then creates and pushes `v<version>`.
3. On GitHub, create a Release from that tag and click **Publish release**
   (or `gh release create v<version> --generate-notes`). Leave "Set as a
   pre-release" unchecked; prereleases are skipped by design.
4. The `release-build` job runs unattended: verifies the tag, lints, tests,
   builds, and checks the artifacts.
5. Approve the `pypi` environment in the Actions run (if reviewers are
   configured) and the upload proceeds.

## What the workflow verifies before uploading

- The release tag equals `v` + the `version` in `pyproject.toml`.
- That version is not already on PyPI (PyPI rejects re-uploads, and it is
  better to find out before the release is public).
- `uv.lock` is in sync with `pyproject.toml` (`uv sync --locked`).
- `ruff check sophon --select E9,F63,F7,F82,F401,F541,F811` passes, the same
  narrow rule set as `ci.yml`.
- The full test suite passes.
- The built wheel and sdist filenames carry the expected version.
- `twine check` passes on both artifacts.

## If something goes wrong

**Tag/version mismatch.** Fix `pyproject.toml` or the tag, delete the release
and tag, and redo. The `published` event does not re-fire when you edit an
existing release; you must publish a new one.

**Version already on PyPI.** Bump the version. PyPI never allows re-uploading a
filename, even after a delete; a bad release can only be *yanked*.

**Publish job stuck on "Waiting for review".** That is the environment gate
working as intended. Approve it from the run page.

## Manual fallback

`make publish` still exists (clean tree → PyPI version check → test → build →
`twine upload` with a token from `~/.pypirc`), as does `make publish-test` for
TestPyPI. Prefer the workflow; the manual path bypasses the reviewer gate.
