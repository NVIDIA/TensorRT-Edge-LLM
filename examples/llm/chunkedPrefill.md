# P3: bounded prompt continuation

`SequenceStepRuntime::beginPrefillChunk(handle)` consumes the next bounded part
of the immutable token vector supplied at admission. The caller formats and
tokenizes the complete prompt once. It must call `completeStep()` before reading
progress, sampling, releasing a slot or starting another forward.

The fixed policy in `runtime/state/prefillChunk.h` uses at most 128 true prompt
tokens per call. Intermediate endpoints remain multiples of 64. When 129–191
tokens remain, it consumes 64 so the final chunk contains 65–127 tokens. Otherwise
it consumes up to 128. Thus a resumed final chunk has 64–128 tokens; a complete
short prompt, including a cold one-token prompt, can be shorter.

Examples:

| Prompt tokens | Chunk spans |
| --- | --- |
| 1 | 1 |
| 65 | 65 |
| 129 | 64, 65 |
| 130 | 64, 66 |
| 191 | 64, 127 |
| 192 | 128, 64 |
| 257 | 128, 64, 65 |
| 6144 | 48 chunks of 128 |

No padding tokens, prompt replay, state-sized copies, new GPU allocation or
sampling occur in this helper. Attention KV, recurrent and convolution storage
continue through the existing selected-row forward mechanism. Each non-final
completion remains in `kPrefill`; only final completion reaches
`kAwaitingSample`. `acceptToken` rejects partial prefill. The first accepted
output is pending input for a later decode; accepting it does not increase the
committed cache length. Sampling/output policy itself remains P5 work.

The fixed policy must be used from the start of prompt processing. It rejects
incompatible manual history (unaligned cursor or a resumed remainder below 64)
before enqueuing GPU work. `beginPrefill(handle, count)` is retained as a low-level
mechanism/diagnostic interface; arbitrary partitions are not numerically
qualified. HTTP integration, automatic scheduling and performance tuning remain
later phases.

## Numerical scope and diagnostics

The original 64+64+1 recipe remains a failing numerical control: its teacher
continuation exceeds the existing maximum absolute 0.1 / relative L2 0.005 logit
screens. The first candidate avoided only singleton tails; expanded tests also
found drift with 2–4-token resumed tails. No threshold was relaxed. The selected
policy avoids these execution shapes while preserving the exact prompt.

On SM87, GDN selects different implementations for physical sequence length one
and greater than one. The prefill implementation is a sequential recurrence;
64-token alignment here is an empirical shape restriction, not a claim that GDN
requires blocks of 64. Kernel/tactic-level attribution of small-shape drift is
not established. Do not extrapolate this policy's qualification to other engines,
precisions, models, devices or chunk caps.

The optional probe has two P3 modes:

- `--chunks`: original boundary controls and longer synthetic prompts through
  6144 tokens, four teacher-forced steps, and all active state before/after decode.
- `--chunks-extra`: every length 129–193, additional multi-chunk tails, four
  once-formatted chat prompts (including code, Unicode/JSON and numbered records),
  eight teacher steps on chats, the retained raw-tail negative control, and
  interleaving in both physical-slot orders.

Logit qualification requires finite values, equal greedy IDs, max absolute error
at most 0.1 and relative L2 at most 0.005. Same-shape short controls and interleaved
references require exact equality. Active-state diagnostics apply finite,
0.1/0.005 screens to each recurrent/conv tensor and each materialized K/V plane;
inactive snapshots require every byte unchanged, including page-table rows.
Unused KV capacity is excluded. Host snapshot/staging copies belong only to the
probe, never to serving execution.

`RAW_TAIL_DIAGNOSTIC quality_passed=0` is expected retained evidence for the
unsupported manual partition; it is separate from `P3_QUALITY_GATE`. Consult the
OpenClaw P3 evidence report for exact tested source revisions, results, failed
attempts, operational restoration, and remaining limitations. A passing short
numerical screen does not establish broad task quality, throughput or long-term
memory stability.
