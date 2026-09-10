# Unit-Testing the Runtime with a Substitute Engine

`LLMInferenceRuntime::handleRequest()` used to be reachable only by building a
TensorRT engine and running `llm_inference`. This guide shows how to drive it
from `./build/unittests/unitTestRuntime` instead, and how to write the same kind of test for
another component.

The worked examples live in `unittests/cpp/runtime/llmInferenceRuntimeAssemblyTests.cpp`.

---

## What is substituted, and what is not

Exactly one collaborator is replaced: `EngineExecutor`. Everything else on the
path is the production object — the parsed deployment config, the KV cache
managers, the pipeline tensors, the tensor map, the decoder registry, the
samplers, and the tokenizer.

That ratio matters. Replace more and the test stops saying anything about the
runtime; the first question in review will be "aren't you just testing your own
fakes?". Keep the substitution at the one boundary that needs a GPU-resident
serialized engine, and everything the test exercises is code that ships.

---

## Getting an object to test

Two things made the runtime constructible without a model directory.

**1. `EngineExecutor` is an interface.** The TensorRT implementation lives
behind it in `engineExecutor.cpp`; `EngineExecutor::createForLLM()` is still the
only way to get one.

The interface kept the original name and every signature, so the 33 files that
reference `EngineExecutor` needed no edits. Prefer this over introducing an
`IFoo` alongside `Foo`: the rename is what turns a two-file change into a
thirty-file one, and a thirty-file diff is much harder to land.

**2. Loading is separated from assembly.** `ModelArtifacts` holds everything the
runtime reads off disk — deployment config, engines, weights, embeddings,
tokenizer. `ModelArtifacts::loadFromEngineDir()` produces one from a directory;
the constructor that takes it does not touch the filesystem.

```cpp
// Production: load, then assemble.
LLMInferenceRuntime runtime{engineDir, "", {}, stream};

// Test: assemble around artifacts nothing loaded.
LLMInferenceRuntime runtime{std::move(artifacts), modelDir, "", {}, std::nullopt, stream};
```

The seam sits on the I/O boundary rather than on an invented abstraction, which
is why it needs no justification in review: "what this class reads from disk" is
an observable fact about the class.

---

## Writing the substitute

`MockEngineExecutor` is a `MOCK_METHOD` declaration per interface method. Wrap it
in `NiceMock` so that assembly-time queries no test cares about stay quiet, and
give those a default with `ON_CALL`:

```cpp
auto engine = std::make_unique<NiceMock<MockEngineExecutor>>();
ON_CALL(*engine, getRequiredContextMemorySize()).WillByDefault(Return(4096));
ON_CALL(*engine, setContextMemory(_)).WillByDefault(Return(true));
```

`NiceMock` silences *uninteresting* calls only. Any method a test puts an
`EXPECT_CALL` on is still checked strictly, so silencing the noise does not
weaken the assertions.

### Leave `getEngine()` without a default

```cpp
//! Returns a reference gmock cannot synthesize, so any call aborts the test.
MOCK_METHOD(nvinfer1::ICudaEngine const&, getEngine, (), (const, noexcept, override));
```

The interface has two kinds of method. `prepare` / `execute` / `captureGraph` /
`setContextMemory` / `setProfiler` / `getRequiredContextMemorySize` run per
request and a substitute must implement them. The introspection accessors run
once during startup validation, and a substitute may refuse them.

Giving `getEngine()` no default action makes that split enforced rather than
documented: if the runtime ever reaches past the interface for the concrete TRT
engine, the test dies at that line instead of quietly passing.

### Standing in for a forward pass

A real engine writes into the tensors it was bound to, so the substitute should
too. Capture the binding in `prepare` and fill it in `execute`:

```cpp
ON_CALL(*engine, prepare(_, _, _, _))
    .WillByDefault([this](int32_t, InferenceDims const&, TensorMap const& map, cudaStream_t) {
        mLogits = map.get(binding_names::kLogits);
        return true;
    });
```

The test then chooses which token each row decodes to, which is what lets it
assert on orchestration instead of on numerics.

---

## Saying what you expect, not counting afterwards

State the calls before the object under test exists:

```cpp
{
    InSequence seq;
    EXPECT_CALL(mock, execute(_)).Times(2).WillRepeatedly(emit({}));
    EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId}));
}
```

A fourth forward now fails on its own, and the failure names the call, the
expectation, and both counts. A counter compared at the end reports `8` against
`3` and leaves the reader to work out what that meant.

### `InSequence` is load-bearing, not decorative

With two expectations on one method, gmock searches them in **reverse
declaration order** and takes the first that matches and is not saturated.
Without `InSequence`, the example above emits EOS on the *first* forward:

```
TOKEN 1
         Expected: to be called twice
           Actual: never called - unsatisfied and active
```

### Partition with matchers instead of ordering

When calls differ by argument, matchers separate them and no sequence is needed:

```cpp
EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(1);
EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(kMaxGenerateLength - 1);
```

This is also how to assert on a struct argument, which is often where the
interesting claim lives:

```cpp
EXPECT_CALL(mock, prepare(kDecodeProfile,
    AllOf(Field(&InferenceDims::seqLen, kVerifySize),
          Field(&InferenceDims::selectLen, kVerifySize)), _, _));
```

Interleaved calls do not belong in one sequence. Prefill and decode alternate
`prepare` then `execute`, so putting both in an `InSequence` would demand
`prepare, prepare, execute, execute`. Sequence the calls that carry information
and assert the counts of the rest separately.

---

## Checking output parameters

`handleRequest` fills a `LLMGenerationResponse&`. Check the API contract as a
block before reading into it — it documents that the per-slot vectors are
repopulated together, to matched sizes:

```cpp
EXPECT_EQ(response.outputIds.size(), slots);
EXPECT_EQ(response.outputTexts.size(), slots);
EXPECT_EQ(response.finishReasons.size(), slots);
EXPECT_EQ(response.inputTokenCounts.size(), slots);
```

Use `ASSERT_` where the next line would otherwise index out of bounds, and
`EXPECT_` elsewhere so one run reports every problem.

Compare containers with matchers rather than a hand-written loop:

```cpp
EXPECT_THAT(response.outputIds[0], AllOf(SizeIs(kMaxGenerateLength), Each(kZeroLogitsToken)));
```

```
Expected: (has a size that is equal to 4) and (only contains elements that is equal to 126)
  Actual: { 127, 127, 127, 127 }, whose element #0 doesn't match
```

---

## Make failures readable

gmock byte-dumps any type it cannot print, which buries the interesting part of
a failed expectation. Add a `PrintTo` in the namespace of the type — ADL finds
it — for anything that appears in a matcher:

```cpp
namespace trt_edgellm { namespace rt {
void PrintTo(Tensor const& tensor, std::ostream* os)
{
    *os << "Tensor{" << tensor.getName() << ", shape=" << tensor.getShape().formatString() << "}";
}
}}
```

Before: `setContextMemory(@0x7ffe 200-byte object <D0-75 3B-27 ...>)`
After: `setContextMemory(@0x7ffe Tensor{LLMInferenceRuntime::mSharedExecContextMemory, shape=[4096]})`

The test file has printers for `Tensor`, `InferenceDims` and `FinishReason`,
each added after a failure proved it was needed.

---

## Two habits worth copying

**Measure the behavior; do not guess it.** The MTP call order and the sampled
token ids were both found by adding a temporary trace, reading the output, then
writing the expectation from it and deleting the trace. Every guess made so far
has been wrong at least once — prefill emits the first token, so N tokens cost N
forwards and not N+1.

**Break each assertion once.** After a test passes, change it so it should fail,
confirm it does, and read the message. This catches assertions that never ran,
and it is how the missing `PrintTo` overloads were found. Commit the test first
so the mutation can be reverted with `git checkout --` rather than by hand.

---

## Constraints the deployment imposes

These bit while building the speculative-decoding tests and are not obvious from
the headers.

| Constraint | Symptom when violated |
|---|---|
| `head_dim` must be 64, 128, 256 or 512 for MTP — it reuses `eagleBaseCommitKVCacheAndAssembleHiddenState` | throws at the first verification round |
| Unmanaged speculative decoding reserves a fixed 100 tokens of KV before admitting generation | request rejected with no forward at all |
| `ExternalWeightManager` runs load → validate → register, each once | `registerTensorMapEntries called before weight validation` |
| `Tokenizer::loadFromHF` needs `processed_chat_template.json` too | load fails |
| `PipelineIO::outputLogits` is FP32, not the engine's compute dtype | plausible-looking wrong argmax, no error |

---

## Seeing runtime debug logs

The test binaries leave the logger at `kINFO`, and there is no environment variable
for it. To see the `LOG_DEBUG` trace the runtime already emits, raise the level
for the call under inspection:

```cpp
gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
```

Reading that trace is usually faster than adding a temporary `printf`, and it
costs nothing to remove afterwards.

---

## Running them

```bash
cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DBUILD_UNIT_TESTS=ON && ninja unitTestRuntime
export LD_LIBRARY_PATH=$TRT_PACKAGE_DIR/lib:$LD_LIBRARY_PATH
./build/unittests/unitTestRuntime --gtest_filter="RuntimeAssemblyTest.*:MtpAssemblyTest.*"
```

`BUILD_UNIT_TESTS=ON` enables gmock. Note that it is set with `FORCE`: a build
tree configured before gmock was enabled already caches `OFF`, and a plain
`set(... CACHE ...)` would leave it there, so the failure would be a missing
`gmock/gmock.h` rather than anything pointing at the cache.
