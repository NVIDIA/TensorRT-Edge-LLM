<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Google documentation practices

Condensed and adapted from Google's
[developer documentation style guide](https://developers.google.com/style/),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Consult the linked pages for details and exceptions.

## Audience, tone, and language

- Write for the reader's task and knowledge. Give enough context to act without
  burying the useful information in introductory filler.
- Use a helpful, conversational, respectful tone. Avoid promotional claims,
  jokes, idioms, and descriptions of a procedure as easy or obvious.
- Address the reader as you. Prefer active voice when identifying the actor
  helps; passive voice can be appropriate when the actor is irrelevant or the
  object deserves emphasis. Describe current behavior in present tense.
- Put conditions before instructions. Use concrete nouns and verbs and resolve
  ambiguous pronouns. Keep terminology consistent and explain unfamiliar
  abbreviations when first needed.
- Use American English unless the project specifies otherwise. Consult the
  word list for usage questions rather than inventing a blanket prohibition.
- Write for international readers and avoid stereotypes, exclusionary language,
  and assumptions about the reader's abilities or background. Preserve exact
  API identifiers even when improving the prose that explains them.

Sources: [voice and tone](https://developers.google.com/style/tone),
[active voice](https://developers.google.com/style/voice),
[present tense](https://developers.google.com/style/tense),
[global audience](https://developers.google.com/style/translation),
[inclusive documentation](https://developers.google.com/style/inclusive-documentation),
[abbreviations](https://developers.google.com/style/abbreviations), and
[word list](https://developers.google.com/style/word-list).

## Organization and procedures

- Use descriptive sentence-case titles and a logical heading hierarchy.
  Prefer imperative task headings for procedures and noun phrases for concepts.
  Preserve proper names and the spelling of literal technical identifiers.
- Put information where readers need it. Introduce a procedure with useful
  context, rather than repeating its heading. Keep prerequisites before steps.
- Number a sequence of steps; use bullets for unordered items. A single-step
  procedure can be a single bullet. Keep list items parallel.
- State the context or condition before the action, and the result after it.
  Mark optional steps clearly. Separate actions when that improves execution,
  and avoid unnecessary nesting or repeated procedures.
- Use tables for comparable data, with meaningful headers. Use prose or lists
  when a table makes reading harder. Avoid merged cells and layout tables.
- Keep essential instructions in the main flow. Use notes for useful asides,
  cautions for care, and warnings for serious or irreversible consequences.
  Avoid stacking notices or presenting routine results as success callouts.

Sources: [headings](https://developers.google.com/style/headings),
[procedures](https://developers.google.com/style/procedures),
[tables](https://developers.google.com/style/tables), and
[notices](https://developers.google.com/style/notices).

## Technical text and examples

- Use code font for literal code, filenames, commands, flags, fields, output,
  and placeholders. Use bold for UI labels and run-in headings, not pervasive
  emphasis. Prefer words that convey importance without extra formatting.
- Describe UI actions by their task and visible label. Avoid identifying a
  control only by its shape, color, or screen position.
- Explain what an example demonstrates before the code block. Follow the
  relevant programming language's style and use the document renderer's
  preformatted code syntax. Do not change executable text to fix prose style.
- Distinguish a runnable example from abbreviated code. Mark omissions with
  language-appropriate comments instead of unexplained ellipses.
- Give placeholders descriptive uppercase names and explain their substitutions
  near the example. Preserve interface syntax where it requires a different
  form. Separate example values, user input, and output clearly.

Sources: [text formatting](https://developers.google.com/style/text-formatting),
[UI elements](https://developers.google.com/style/ui-elements),
[code samples](https://developers.google.com/style/code-samples), and
[placeholders](https://developers.google.com/style/placeholders).

## Punctuation, numbers, and terminology

Use serial commas and consistent American punctuation. Distinguish punctuation
in prose from literal code. Consult the specific punctuation topic when needed;
Google's guide does not ban all em dashes or all passive constructions.

Make quantities, units, ranges, dates, and times unambiguous. Keep unit symbols
consistent and distinguish decimal and binary storage quantities where relevant.
Use dates that international readers can interpret and specify time zones when
they matter. Preserve exact product names, version strings, and interface names.

Sources: [highlights](https://developers.google.com/style/highlights),
[dashes](https://developers.google.com/style/dashes),
[numbers](https://developers.google.com/style/numbers),
[units](https://developers.google.com/style/units-of-measure),
[dates and times](https://developers.google.com/style/dates-times), and
[word list](https://developers.google.com/style/word-list).

## Links, images, and accessibility

- Make link text identify the destination or purpose when read independently.
  Avoid vague labels such as here. Explain unexpected downloads or other link
  behavior. Keep essential context on the page rather than hiding it in a link.
- Use semantic headings, lists, and tables. Refer to sections or controls by
  name rather than relying on visual position.
- Give informative images meaningful alternative text and explain diagrams in
  surrounding prose. Decorative images can have empty alternative text.
  Prefer text to screenshots of code or terminal output.
- Do not encode meaning through color or icons alone. Provide textual
  equivalents for visual information and captions or transcripts for media.
- When editing rendered interactions, check keyboard access and reading order.
  Preserve the site's accessible components instead of adding custom styling
  merely to reproduce Google's site appearance.

Sources: [cross-references](https://developers.google.com/style/cross-references)
and [accessibility](https://developers.google.com/style/accessibility).
