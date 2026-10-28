# Guided Decoding

## Overview

Guided decoding constrains generation so the output is *guaranteed* to match a schema, pattern, or grammar. At every decoding step the runtime asks a grammar engine (XGrammar) which tokens are still legal and drives the logits of every other token to a large negative value before sampling. Illegal tokens therefore cannot be sampled at all — this is a hard constraint, not a prompt-level suggestion or a retry loop.

**Key Points**:
- Six modes: `json_object`, `json_schema`, `regex`, `ebnf`, `structural_tag`, `choice`
- Two request surfaces: the low-level `guided_decoding` field, and the OpenAI-compatible `response_format` on the server
- Per-request and per-slot: constrained and unconstrained requests can share a batch
- Compiled grammars are cached, so repeating a schema across requests costs one lookup

---

## Build

Guided decoding is built into every runtime; there is nothing to enable. The XGrammar backend is a Git submodule that CMake checks out on demand, so a normal build needs no extra steps. If your environment has no network access at configure time, check it out yourself first:

```bash
git submodule update --init --depth 1 --checkout 3rdParty/xgrammar
git -C 3rdParty/xgrammar submodule update --init --depth 1 --checkout 3rdparty/dlpack
```

---

## Usage

### Request file

Set exactly one mode under `guided_decoding`. It can be set per request, or at the top level of the file as a default for every request.

**`json_schema`** — the most common mode. Output is a JSON document matching the schema:

```json
{
    "requests": [
        {
            "messages": [{"role": "user", "content": "Give me a person record."}],
            "guided_decoding": {
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
                }
            }
        }
    ]
}
```

The schema may be given as an inline object (above) or as an escaped JSON string.

**`json_object`** — any JSON object, no schema:

```json
{"guided_decoding": {"json_object": true}}
```

**`regex`** — the whole output matches the pattern:

```json
{"guided_decoding": {"regex": "[a-z]+@[a-z]+\\.(com|org)"}}
```

**`ebnf`** — a GBNF-style grammar. It must define a rule named `root`:

```json
{"guided_decoding": {"ebnf": "root ::= \"yes\" | \"no\""}}
```

**`choice`** — output is exactly one of the listed strings. A convenience form of `ebnf`; the alternatives are escaped for you:

```json
{"guided_decoding": {"choice": ["yes", "no", "maybe"]}}
```

**`structural_tag`** — free text until a trigger appears, then the tagged region is constrained. Used for tool calling, where only the arguments need to be well-formed:

```json
{
    "guided_decoding": {
        "structural_tag": {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=get_weather>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"]
                            }
                        },
                        "end": "</function>"
                    }
                ]
            }
        }
    }
}
```

### OpenAI-compatible server

The server accepts both surfaces. Use `response_format` for portability with OpenAI clients:

```json
{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Give me a person record."}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        }
    }
}
```

`response_format` covers only `text` (unconstrained), `json_object`, and `json_schema` — that is the whole of the OpenAI specification. For `regex`, `ebnf`, `structural_tag`, and `choice`, use the low-level field, which takes the same shape as in a request file:

```json
{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Answer yes or no."}],
    "guided_decoding": {"choice": ["yes", "no"]}
}
```

Setting both surfaces on one request is an error rather than one silently winning. Streaming works normally.

---

## Schema Support

XGrammar enforces most of JSON Schema, but a few keywords are *accepted and then ignored*. Since silently returning output that violates the schema is worse than refusing the request, these are rejected up front with a message naming the keyword:

`multipleOf`, `uniqueItems`, `contains`, `minContains`, `maxContains`, `patternProperties`

`format` is enforced only for the values XGrammar implements: `date`, `date-time`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `json-pointer`, `relative-json-pointer`, `time`, `uri`, `uri-reference`, `uri-template`, `uuid`. Any other `format` value is rejected for the same reason.

The check only inspects schema keywords, so a property genuinely *named* `contains` is fine.

---

## Speculative decoding

Every speculative decoding mode is supported, with no request-side configuration.

| Mode | Geometry |
|---|---|
| MTP | chain and tree |
| EAGLE3 | tree |
| DFlash | linear and DDTree |
| DFlash2 | chain |
| JetSpec | DDTree |
| DSpark | chain and tree, greedy or sampled |
| Gemma4 MTP | chain |

A speculative step verifies several candidate tokens at once, and each verification row sits at a
different point in the grammar, so the mask is built per row rather than per request: the decoder
hands the shape of its draft tree to the guided decoder, which walks the grammar down the tree and
masks every row from the state its own path reaches. A candidate the grammar refuses takes its
subtree with it, leaving those rows unmasked; they are unreachable anyway, because acceptance
follows a root-to-node path.

Constrained output stays valid whether or not drafting is on, but it is not always *identical*.
Tree geometries verify each node under an attention mask covering only its own path, which differs
numerically from the vanilla forward, so a near-tie can resolve differently and long unconstrained
stretches may diverge. Chain geometries do not have this effect.

## Limitations

- **Not supported on block-diffusion engines**, which denoise a whole canvas per step instead of appending one token at a time. Such requests are rejected rather than silently left unconstrained.
- **Thinking models**: the constraint starts at the token *immediately after* the reasoning-end
  marker (`</think>` or `<channel|>`), including when speculative decoding commits the marker and
  the tokens following it in the same step. The reasoning block itself is unconstrained. A request
  whose reasoning never ends is therefore never constrained.

  Whether a request starts inside a reasoning block is read from the rendered prompt. The most
  recent reasoning marker is authoritative. If the prompt has no marker, `enable_thinking=false`
  starts the constraint immediately, while `enable_thinking=true` leaves room for the model to
  open a reasoning block. Models without reasoning-marker tokens are constrained immediately.
- **Regex dialect**: XGrammar's regex, not PCRE. No lookaround and no backreferences.
- **EBNF entry rule** must be named `root`.
- **Guide size** is capped at 128 KB. Compilation is superlinear in guide size and runs before any GPU work.
- **`structural_tag`** accepts only the modern `triggered_tags` form, not the legacy `{structures, triggers}` shape.

---

## Notes

- A guide that fails to compile fails only its own request; the rest of the batch continues.
- If a grammar reaches a state where no token in the model's vocabulary is legal, the request ends with `finish_reason: error` and keeps the tokens generated so far, which do *not* satisfy the guide. When streaming, those tokens have already been sent, so check `finish_reason` before using the result. This normally means the guide cannot be expressed in this tokenizer's vocabulary.
- Constrained requests are slightly slower per step (one mask fill plus one elementwise kernel), but usually *finish sooner*, because the model cannot wander outside the grammar.
