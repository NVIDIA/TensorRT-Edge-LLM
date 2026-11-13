---
name: edge-llm-documentation
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
description: Write, edit, or review TensorRT Edge-LLM documentation using the Google developer documentation style guide. Apply to READMEs, tutorials, how-to guides, conceptual and design documents, API and CLI references, and release notes; not ordinary code changes or chat replies.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Edge-LLM documentation

Use the [Google developer documentation style guide](https://developers.google.com/style/)
as the editorial reference for the whole document, including organization,
language, formatting, links, accessibility, and technical examples.
Read the bundled [Google documentation practices](references/google-documentation-practices.md)
before writing or reviewing; this summary is sufficient for routine work.
Consult individual Google pages only for uncovered topics or ambiguities;
there is no need to reread the entire guide for each task.

## Establish the scope and sources

Read the requested document, nearby pages, and applicable repository instructions.
Identify the audience, reader's task, target release or branch, and output format.
Use the user's explicit requirements first, then explicit project conventions,
then Google's guide. Existing prose can contain mistakes; do not treat every
existing spelling or layout as a required convention.

If a needed source cannot be accessed, state the verification gap rather than inventing a
Google rule. Do not apply Android- or Google Cloud-specific conventions to
Edge-LLM without a relevant reason.

For product facts, inspect branch-local implementation, command help, build
configuration, tests, and maintained documentation. Google's guide determines
editorial choices, not supported models, platforms, defaults, or behavior.

## Write or revise

Choose a structure suited to the request: a learning sequence for a tutorial,
an actionable procedure for a how-to, an explanation for a concept, or precise
lookup information for a reference. Keep a narrow correction narrow. For a
review-only request, report findings without editing files.

Apply the reference across the requested scope. Preserve technical meaning,
literal identifiers, and prerequisites while improving prose. Explain proposed
behavior as proposed in design documents; describe implemented behavior in user
guides. Do not convert every document into a tutorial or force a fixed template.

## Apply Edge-LLM conventions

- Use Markdown for new prose documents unless the user or destination's
  conventions specify another format. Use Jupyter notebooks only for explicitly
  requested executable tutorials.
- Preserve the document's Markdown, reStructuredText, or AsciiDoc format and
  supported directives. Keep Sphinx cross-references, explicit anchors, and
  navigation working when headings or paths change.
- For work under `design/`, read [the design index](../../design/README.adoc)
  and [design contribution guidance](../../design/CONTRIBUTING.adoc). Follow
  their document structure and AsciiDoc conventions, and verify technical claims
  against current branch code and tests.
- Read `docs/README.md` and `docs/source/conf.py` before changing site syntax or
  building the site. Use the existing renderer's code blocks and admonitions;
  do not transplant Google's website-specific HTML or CSS.
- Identify generated documentation before editing. For generated API pages,
  update the owning Python docstring, Doxygen comment, or generator, then
  regenerate the output. Direct edits to generated pages can disappear during
  the next documentation build.
- Follow `CODING_GUIDELINES.md` for code examples and Python docstring syntax.
  Google's documentation prose guidance does not replace C++ or Python rules.
- Verify commands, flags, paths, defaults, and dependencies against the target
  branch. Separate host setup from device execution and identify relevant
  platform, precision, and version constraints. Do not combine distinct export
  or runtime routes into an apparently interchangeable procedure.
- Keep documented benchmark results attached to their model, hardware,
  precision, workload, and measurement conditions. Do not turn an observed
  result into a general performance or support guarantee.
- Public documentation must be usable from the public tree. Preserve release
  exclusion markers and avoid introducing internal-only links or prerequisites
  into public instructions. Link to a maintained procedure instead of copying
  commands that can diverge.

## Verify and deliver

Review factual correctness separately from editorial quality. Check local links,
anchors, navigation, code block syntax, and placeholders. For site changes, run
the documented build when dependencies are available and inspect affected
rendered pages. Run applicable pre-commit hooks on changed files.

Validate changed examples when the environment permits. Distinguish static checks
from execution; a documentation build does not prove that a model
works. A model-validation claim requires export, engine build, and inference
under the repository's validation policy.

For a style review, give the location, reader impact, suggested correction, and
the relevant Google source or project convention. Distinguish factual defects
from editorial suggestions; preserve justified exceptions that improve clarity.

Apply edits directly to the target document. Write for its readers without
recapping the prompt, conversation, or editing process. Do not add sections that
summarize or justify the editing process unless explicitly requested. Keep
content the document's reader needs, such as architectural rationale or steps
to verify a tutorial's result. Preserve useful technical context and attribution;
omit claims about excluded approaches that merely repeat a conversational
constraint. Do not remove existing sections or restructure unrelated content
solely to apply this guidance.

For editing requests, return only requested file changes or a commit message.
If an external edit needs confirmation, give a brief confirmation or artifact
link without a recap. Still perform relevant checks and disclose actual blockers
or failed actions.
