/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Exhaustive unit tests for the streaming infrastructure introduced in the
// llm-streaming implementation plan. These tests exercise the parts that are
// pure CPU / no-GPU:
//   - cpp/common/utf8.{h,cpp}               (UTF-8 sanitizer + flush)
//   - cpp/tokenizer/tokenizer.{h,cpp}       (idToPiece, emitDelta, emitDeltaFlush,
//                                           Tokenizer::decode sanitization)
//   - cpp/runtime/llmRuntimeUtils.{h,cpp}   (StreamChannel, StreamChunk,
//                                           SlotStreamState, compactVector)
//
// The live-runtime integration scenarios (B*, S*, C*, L* from the plan) run
// against real engines via examples/llm/llm_stream — see that binary.
//

#include "common/utf8.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenEncoder.h"
#include "tokenizer/tokenizer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace trt_edgellm;
using trt_edgellm::rt::FinishReason;
using trt_edgellm::rt::SlotStreamState;
using trt_edgellm::rt::StreamChannel;
using trt_edgellm::rt::StreamChunk;

namespace
{
constexpr char const* kFFFD = "\xEF\xBF\xBD";
constexpr size_t kFFFDLen = 3;
} // namespace

TEST(TokenizerTest, SetAdditionalEosIdsFromEngineConfig)
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_tokenizer_multi_eos_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {"a": 0, "<eos>": 1, "<bos>": 2, "<turn|>": 106}, "merges": []},
  "added_tokens": [
    {"id": 1, "content": "<eos>"},
    {"id": 2, "content": "<bos>"},
    {"id": 106, "content": "<turn|>"}
  ],
  "pre_tokenizer": {"type": "Split", "pattern": {"String": ""}}
})JSON";
    std::ofstream(dir / "tokenizer_config.json")
        << R"JSON({"eos_token": {"content": "<eos>"}, "bos_token": {"content": "<bos>"}})JSON";
    std::ofstream(dir / "processed_chat_template.json") << R"JSON({
  "model_path": "unit",
  "roles": {
    "system": {"prefix": "", "suffix": ""},
    "user": {"prefix": "", "suffix": ""},
    "assistant": {"prefix": "", "suffix": ""}
  },
  "generation_prompt": ""
})JSON";

    tokenizer::Tokenizer tokenizer;
    ASSERT_TRUE(tokenizer.loadFromHF(dir));
    tokenizer.setAdditionalEosIds({1, 106, 106});
    EXPECT_EQ(tokenizer.getEosId(), 1);
    EXPECT_TRUE(tokenizer.isEosId(1));
    EXPECT_TRUE(tokenizer.isEosId(106));
    EXPECT_FALSE(tokenizer.isEosId(7));
    EXPECT_EQ(tokenizer.getEosIds(), (std::vector<tokenizer::Rank>{1, 106}));

    std::filesystem::remove_all(dir);
}

TEST(TokenizerTest, AppliesThinkingChatTemplateFields)
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_tokenizer_thinking_template_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "processed_chat_template.json") << R"JSON({
  "model_path": "unit",
  "roles": {
    "system": {
      "prefix": "<bos><|turn>system\n",
      "suffix": "<turn|>\n",
      "prefix_thinking": "<bos><|turn>system\n<|think|>\n",
      "suffix_thinking": "<turn|>\n"
    },
    "user": {"prefix": "<bos><|turn>user\n", "suffix": "<turn|>\n"},
    "assistant": {"prefix": "<|turn>model\n", "suffix": "<turn|>\n"}
  },
  "generation_prompt": "<|turn>model\n<|channel>thought\n<channel|>",
  "generation_prompt_thinking": "<|turn>model\n"
})JSON";

    tokenizer::Tokenizer tokenizer;
    ASSERT_TRUE(tokenizer.loadChatTemplate(dir / "processed_chat_template.json"));

    rt::LLMGenerationRequest::Request request;
    request.messages = {
        rt::Message{"system", {rt::Message::MessageContent{"text", "Reply with only the letter."}}},
        rt::Message{"user", {rt::Message::MessageContent{"text", "Question?\nAnswer: "}}},
    };

    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(tokenizer.applyChatTemplate(
        request, formatted, /*applyChatTemplate=*/true, /*addGenerationPrompt=*/true, /*enableThinking=*/false));
    EXPECT_EQ(formatted.formattedCompleteRequest,
        "<bos><|turn>system\nReply with only the letter.<turn|>\n"
        "<bos><|turn>user\nQuestion?\nAnswer: <turn|>\n"
        "<|turn>model\n<|channel>thought\n<channel|>");

    ASSERT_TRUE(tokenizer.applyChatTemplate(
        request, formatted, /*applyChatTemplate=*/true, /*addGenerationPrompt=*/true, /*enableThinking=*/true));
    EXPECT_EQ(formatted.formattedCompleteRequest,
        "<bos><|turn>system\n<|think|>\nReply with only the letter.<turn|>\n"
        "<bos><|turn>user\nQuestion?\nAnswer: <turn|>\n"
        "<|turn>model\n");

    std::filesystem::remove_all(dir);
}

TEST(TokenizerTest, AppliesTrimContentChatTemplateFields)
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_tokenizer_trim_template_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "processed_chat_template.json") << R"JSON({
  "model_path": "unit",
  "roles": {
    "system": {"prefix": "<bos><|turn>system\n", "suffix": "<turn|>\n", "trim_content": true},
    "user": {"prefix": "<|turn>user\n", "suffix": "<turn|>\n", "trim_content": true},
    "assistant": {"prefix": "<|turn>model\n", "suffix": "<turn|>\n", "trim_content": true}
  },
  "generation_prompt": "<|turn>model\n"
})JSON";

    tokenizer::Tokenizer tokenizer;
    ASSERT_TRUE(tokenizer.loadChatTemplate(dir / "processed_chat_template.json"));

    rt::LLMGenerationRequest::Request request;
    request.messages = {
        rt::Message{"system", {rt::Message::MessageContent{"text", "  Reply with only the letter.  "}}},
        rt::Message{"user", {rt::Message::MessageContent{"text", "  Question?\nAnswer:  "}}},
    };

    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(tokenizer.applyChatTemplate(
        request, formatted, /*applyChatTemplate=*/true, /*addGenerationPrompt=*/true, /*enableThinking=*/false));
    EXPECT_EQ(formatted.formattedCompleteRequest,
        "<bos><|turn>system\nReply with only the letter.<turn|>\n"
        "<|turn>user\nQuestion?\nAnswer:<turn|>\n"
        "<|turn>model\n");

    std::filesystem::remove_all(dir);
}

TEST(TokenizerTest, AppliesGlobalPrefixOnce)
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_tokenizer_global_prefix_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "processed_chat_template.json") << R"JSON({
  "model_path": "unit",
  "global_prefix": "<bos>",
  "roles": {
    "system": {"prefix": "<|turn>system\n", "suffix": "<turn|>\n"},
    "user": {"prefix": "<|turn>user\n", "suffix": "<turn|>\n"},
    "assistant": {"prefix": "<|turn>model\n", "suffix": "<turn|>\n"}
  },
  "generation_prompt": "<|turn>model\n"
})JSON";

    tokenizer::Tokenizer tokenizer;
    ASSERT_TRUE(tokenizer.loadChatTemplate(dir / "processed_chat_template.json"));

    rt::LLMGenerationRequest::Request userOnlyRequest;
    userOnlyRequest.messages = {
        rt::Message{"user", {rt::Message::MessageContent{"text", "Question?"}}},
    };

    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(tokenizer.applyChatTemplate(userOnlyRequest, formatted, /*applyChatTemplate=*/true,
        /*addGenerationPrompt=*/true, /*enableThinking=*/false));
    EXPECT_EQ(formatted.formattedCompleteRequest, "<bos><|turn>user\nQuestion?<turn|>\n<|turn>model\n");
    EXPECT_EQ(formatted.formattedSystemPrompt, "");

    rt::LLMGenerationRequest::Request systemRequest;
    systemRequest.messages = {
        rt::Message{"system", {rt::Message::MessageContent{"text", "System."}}},
        rt::Message{"user", {rt::Message::MessageContent{"text", "Question?"}}},
    };

    ASSERT_TRUE(tokenizer.applyChatTemplate(systemRequest, formatted, /*applyChatTemplate=*/true,
        /*addGenerationPrompt=*/true, /*enableThinking=*/false));
    EXPECT_EQ(formatted.formattedSystemPrompt, "<bos><|turn>system\nSystem.<turn|>\n");
    EXPECT_EQ(formatted.formattedCompleteRequest,
        "<bos><|turn>system\nSystem.<turn|>\n"
        "<|turn>user\nQuestion?<turn|>\n"
        "<|turn>model\n");

    std::filesystem::remove_all(dir);
}

TEST(TokenizerTest, RawPromptModePreservesWhitespaceWithTrimTemplate)
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_tokenizer_raw_prompt_trim_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "processed_chat_template.json") << R"JSON({
  "model_path": "unit",
  "roles": {
    "user": {"prefix": "<|turn>user\n", "suffix": "<turn|>\n", "trim_content": true}
  },
  "generation_prompt": "<|turn>model\n"
})JSON";

    tokenizer::Tokenizer tokenizer;
    ASSERT_TRUE(tokenizer.loadChatTemplate(dir / "processed_chat_template.json"));

    rt::LLMGenerationRequest::Request request;
    request.messages = {
        rt::Message{"user", {rt::Message::MessageContent{"text", "  raw prompt<|turn>model\n"}}},
    };

    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(tokenizer.applyChatTemplate(
        request, formatted, /*applyChatTemplate=*/false, /*addGenerationPrompt=*/false, /*enableThinking=*/false));
    EXPECT_EQ(formatted.formattedCompleteRequest, "  raw prompt<|turn>model\n");

    std::filesystem::remove_all(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// UTF-8 Sanitizer Tests
// ─────────────────────────────────────────────────────────────────────────────

class Utf8SanitizerTest : public ::testing::Test
{
};

TEST_F(Utf8SanitizerTest, EmptyInputEmptyPending)
{
    std::string pending;
    EXPECT_EQ(utf8::sanitizeUtf8Streaming("", pending), "");
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, PureAsciiPassesThroughUnchanged)
{
    std::string pending;
    std::string const in = "Hello, world!";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), in);
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, TwoByteCodepointEmitted)
{
    std::string pending;
    // U+00E9 LATIN SMALL LETTER E WITH ACUTE ("é") = 0xC3 0xA9
    std::string const in = "\xC3\xA9";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), in);
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, ThreeByteCJKEmitted)
{
    std::string pending;
    // U+4E2D CJK UNIFIED IDEOGRAPH "中" = 0xE4 0xB8 0xAD
    std::string const in = "\xE4\xB8\xAD";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), in);
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, FourByteEmojiEmitted)
{
    std::string pending;
    // U+1F999 "🦙" LLAMA = 0xF0 0x9F 0xA6 0x99
    std::string const in = "\xF0\x9F\xA6\x99";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), in);
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, TrailingIncompleteLeaderHeldInPending)
{
    std::string pending;
    // 0xF0 is a 4-byte leader; feed only the first byte.
    std::string const in = "\xF0";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), "");
    EXPECT_EQ(pending, "\xF0");
}

TEST_F(Utf8SanitizerTest, TrailingIncompleteTwoBytesHeldInPending)
{
    std::string pending;
    std::string const in = "\xF0\x9F";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), "");
    EXPECT_EQ(pending, "\xF0\x9F");
}

TEST_F(Utf8SanitizerTest, PendingPrependedOnNextCall)
{
    std::string pending;
    // Split 🦙 (0xF0 0x9F 0xA6 0x99) across three calls.
    EXPECT_EQ(utf8::sanitizeUtf8Streaming("\xF0\x9F", pending), "");
    EXPECT_EQ(pending, "\xF0\x9F");
    EXPECT_EQ(utf8::sanitizeUtf8Streaming("\xA6", pending), "");
    EXPECT_EQ(pending, "\xF0\x9F\xA6");
    EXPECT_EQ(utf8::sanitizeUtf8Streaming("\x99", pending), "\xF0\x9F\xA6\x99");
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, IsolatedContinuationByteIsReplaced)
{
    std::string pending;
    // 0x80 alone — illegal leader byte.
    std::string const in
        = "A\x80"
          "B";
    std::string const expected = std::string("A") + kFFFD + "B";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), expected);
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, OverlongTwoByteRejected)
{
    std::string pending;
    // 0xC0 0xAF would overlong-encode '/', rejected.
    std::string const in = "\xC0\xAF";
    std::string const out = utf8::sanitizeUtf8Streaming(in, pending);
    // 0xC0 is an invalid leader (overlong), so treated as invalid byte → U+FFFD,
    // then 0xAF is isolated continuation → another U+FFFD.
    EXPECT_EQ(out, std::string(kFFFD) + kFFFD);
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, FiveByteLeaderRejected)
{
    std::string pending;
    // 0xF8 is not a valid leader in UTF-8.
    std::string const in
        = "A\xF8"
          "B";
    std::string const expected = std::string("A") + kFFFD + "B";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), expected);
}

TEST_F(Utf8SanitizerTest, SurrogateRejected)
{
    std::string pending;
    // 0xED 0xA0 0x80 encodes U+D800 surrogate — rejected.
    std::string const in = "\xED\xA0\x80";
    std::string const out = utf8::sanitizeUtf8Streaming(in, pending);
    EXPECT_EQ(out.substr(0, kFFFDLen), kFFFD);
}

TEST_F(Utf8SanitizerTest, ContinuationWithoutLeaderReplaced)
{
    std::string pending;
    // Valid leader missing continuation: 0xC3 alone then next call has bogus
    // leader.
    std::string const in = "\xC3\x41"; // 0x41 is 'A' — NOT a continuation.
    std::string const out = utf8::sanitizeUtf8Streaming(in, pending);
    // 0xC3 expects 1 continuation; 0x41 fails the check → 0xC3 replaced.
    // Then 'A' passes as ASCII.
    EXPECT_EQ(out, std::string(kFFFD) + "A");
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, MixedValidAndInvalidBytes)
{
    std::string pending;
    std::string const in = "He\x80llo\xFF!";
    std::string const expected = std::string("He") + kFFFD + "llo" + kFFFD + "!";
    EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, pending), expected);
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, FlushHeldBytesToReplacementChars)
{
    std::string pending = "\xF0\x9F\xA6"; // 3 held bytes of an incomplete 🦙
    std::string const out = utf8::sanitizeUtf8Flush(pending);
    EXPECT_EQ(out, std::string(kFFFD) + kFFFD + kFFFD);
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, FlushOnEmptyPendingIsNoop)
{
    std::string pending;
    EXPECT_EQ(utf8::sanitizeUtf8Flush(pending), "");
    EXPECT_TRUE(pending.empty());
}

TEST_F(Utf8SanitizerTest, AllSingleByteAsciiSurviveUnchanged)
{
    std::string pending;
    for (unsigned char b = 0x01; b < 0x80; ++b)
    {
        std::string const in(1, static_cast<char>(b));
        std::string p;
        EXPECT_EQ(utf8::sanitizeUtf8Streaming(in, p), in)
            << "ASCII byte 0x" << std::hex << static_cast<int>(b) << " not preserved";
        EXPECT_TRUE(p.empty());
    }
}

TEST_F(Utf8SanitizerTest, AllIsolatedHighBytesReplaced)
{
    for (int bi = 0x80; bi < 0x100; ++bi)
    {
        auto const b = static_cast<unsigned char>(bi);
        // Skip leaders that happen to be valid but just incomplete (0xC0-0xF7 family).
        std::string pending;
        std::string const in(1, static_cast<char>(b));
        std::string const out = utf8::sanitizeUtf8Streaming(in, pending);
        if (pending.empty())
        {
            // Byte was treated as invalid leader or isolated continuation —
            // must have emitted U+FFFD.
            EXPECT_EQ(out, kFFFD) << "Expected U+FFFD for isolated byte 0x" << std::hex << static_cast<int>(b);
        }
        else
        {
            // Held as pending (valid incomplete leader).
            EXPECT_TRUE(out.empty());
            EXPECT_EQ(pending.size(), 1U);
            // Flush must then turn the held byte into U+FFFD.
            std::string const flushed = utf8::sanitizeUtf8Flush(pending);
            EXPECT_EQ(flushed, kFFFD);
            EXPECT_TRUE(pending.empty());
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamChannel — single-consumer tests
// ─────────────────────────────────────────────────────────────────────────────

// Test fixture exposes the friend producer-side API by declaring itself a
// friend of StreamChannel. We can't, so we instead create a tiny "runtime-like"
// friend shim by adding `StreamChannelFinalizer` access OR by exercising the
// public paths through the runtime's friend methods. Here we go the simpler
// route: most producer-facing behavior is indirectly exercised through the
// runtime and finalizer code paths (other test sections). For direct push
// testing we leverage StreamChannelFinalizer via a minimally-populated context.
//
// Below we cover the consumer-facing guarantees (consume/tryPop/waitPop/cancel/
// setStreamInterval/isFinished/getReason/getOriginalBatchIdx) whose behavior is
// observable without producer access, plus cross-thread interaction via std::thread.

class StreamChannelTest : public ::testing::Test
{
};

TEST_F(StreamChannelTest, FactoryCreatesDistinctSharedPtrs)
{
    auto a = StreamChannel::create();
    auto b = StreamChannel::create();
    EXPECT_NE(a.get(), b.get());
    EXPECT_EQ(a.use_count(), 1);
    EXPECT_EQ(b.use_count(), 1);
}

TEST_F(StreamChannelTest, InitialStateIsClean)
{
    auto ch = StreamChannel::create();
    EXPECT_FALSE(ch->isFinished());
    EXPECT_FALSE(ch->isCancelled());
    EXPECT_EQ(ch->getReason(), FinishReason::kNotFinished);
    EXPECT_EQ(ch->getOriginalBatchIdx(), -1);
    EXPECT_EQ(ch->getStreamInterval(), 1);
}

TEST_F(StreamChannelTest, TryPopOnEmptyReturnsNullopt)
{
    auto ch = StreamChannel::create();
    EXPECT_FALSE(ch->tryPop().has_value());
}

TEST_F(StreamChannelTest, WaitPopTimesOutOnEmpty)
{
    auto ch = StreamChannel::create();
    auto const t0 = std::chrono::steady_clock::now();
    auto r = ch->waitPop(std::chrono::milliseconds{30});
    auto const elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_FALSE(r.has_value());
    EXPECT_GE(elapsed, std::chrono::milliseconds{25});
}

TEST_F(StreamChannelTest, CancelWakesWaitPop)
{
    auto ch = StreamChannel::create();
    std::atomic<bool> done{false};
    std::thread t([&] {
        (void) ch->waitPop(std::chrono::seconds{5});
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    EXPECT_FALSE(done.load());
    ch->cancel();
    t.join();
    EXPECT_TRUE(done.load());
    EXPECT_TRUE(ch->isCancelled());
}

TEST_F(StreamChannelTest, CancelWakesConsume)
{
    auto ch = StreamChannel::create();
    std::atomic<bool> done{false};
    std::thread t([&] {
        ch->consume([](StreamChunk&&) {}, std::chrono::milliseconds{10});
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    EXPECT_FALSE(done.load());
    ch->cancel();
    t.join();
    EXPECT_TRUE(done.load());
}

TEST_F(StreamChannelTest, StreamIntervalClampedAtLeastOne)
{
    auto ch = StreamChannel::create();
    ch->setStreamInterval(0);
    EXPECT_EQ(ch->getStreamInterval(), 1);
    ch->setStreamInterval(-5);
    EXPECT_EQ(ch->getStreamInterval(), 1);
    ch->setStreamInterval(8);
    EXPECT_EQ(ch->getStreamInterval(), 8);
}

TEST_F(StreamChannelTest, SkipSpecialTokensDefaultsTrue)
{
    auto ch = StreamChannel::create();
    EXPECT_TRUE(ch->getSkipSpecialTokens());
    ch->setSkipSpecialTokens(false);
    EXPECT_FALSE(ch->getSkipSpecialTokens());
    ch->setSkipSpecialTokens(true);
    EXPECT_TRUE(ch->getSkipSpecialTokens());
}

TEST_F(StreamChannelTest, DoubleCancelIsIdempotent)
{
    auto ch = StreamChannel::create();
    ch->cancel();
    ch->cancel();
    ch->cancel();
    EXPECT_TRUE(ch->isCancelled());
    // Still safe to query terminal state.
    EXPECT_FALSE(ch->isFinished()); // cancel alone does not mark finished
}

TEST_F(StreamChannelTest, CancelledChannelIsNotFinishedWithoutFinishCall)
{
    auto ch = StreamChannel::create();
    ch->cancel();
    EXPECT_TRUE(ch->isCancelled());
    EXPECT_FALSE(ch->isFinished());
    EXPECT_EQ(ch->getReason(), FinishReason::kNotFinished);
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamChannel — producer-side + consumer interaction via
// StreamChannelFinalizer (which is friend of StreamChannel, so it is the
// officially-supported test harness for producer APIs).
// ─────────────────────────────────────────────────────────────────────────────

namespace
{
// Drive the finalizer to exercise push() + finish() on a channel.
// Builds a DecodingInferenceContext whose only slotStream carries an aliased
// shared_ptr to the test-owned channel (no ownership transfer). The finalizer
// sees tokenIds empty, so it pushes a terminal chunk with no tokens/text and
// reason=kError, then calls finish(kError).
void finalizeEmpty(std::shared_ptr<StreamChannel> const& ch)
{
    rt::DecodingInferenceContext ctx;
    ctx.slotStreams.resize(1);
    ctx.slotStreams[0].channel = ch;
    tokenizer::Tokenizer tok; // unloaded — idToPiece returns "" and will not be called anyway
    {
        rt::StreamChannelFinalizer f(ctx, tok);
    }
}
} // namespace

TEST_F(StreamChannelTest, FinalizerTerminalChunkDeliveredToConsume)
{
    auto ch = StreamChannel::create();
    std::vector<StreamChunk> chunks;
    std::thread consumer([&] { ch->consume([&](StreamChunk&& c) { chunks.push_back(std::move(c)); }); });

    finalizeEmpty(ch);

    consumer.join();
    ASSERT_EQ(chunks.size(), 1U);
    EXPECT_TRUE(chunks[0].finished);
    EXPECT_EQ(chunks[0].reason, FinishReason::kError);
    EXPECT_TRUE(chunks[0].tokenIds.empty());
    EXPECT_TRUE(chunks[0].text.empty());
    EXPECT_TRUE(ch->isFinished());
    EXPECT_EQ(ch->getReason(), FinishReason::kError);
}

TEST_F(StreamChannelTest, FinishIsIdempotentFirstWriterWins)
{
    auto ch = StreamChannel::create();
    // Run finalizer twice (second call will observe isFinished and skip).
    finalizeEmpty(ch);
    EXPECT_TRUE(ch->isFinished());
    EXPECT_EQ(ch->getReason(), FinishReason::kError);

    // Finalize a fresh context pointing at the same channel — it should skip
    // pushing another terminal chunk because isFinished() is true.
    {
        rt::DecodingInferenceContext ctx;
        ctx.slotStreams.resize(1);
        ctx.slotStreams[0].channel = ch;
        tokenizer::Tokenizer tok;
        rt::StreamChannelFinalizer f(ctx, tok);
    }
    // Consume what's in the queue and make sure there's still exactly one chunk.
    int count = 0;
    while (auto c = ch->tryPop())
    {
        (void) c;
        count++;
    }
    EXPECT_EQ(count, 1);
    EXPECT_EQ(ch->getReason(), FinishReason::kError);
}

TEST_F(StreamChannelTest, ConsumeDeliversAllChunksBeforeExit)
{
    auto ch = StreamChannel::create();

    // Use a context with multiple slotStreams pointing at the same channel —
    // this causes the finalizer to push multiple terminal chunks.
    // But the plan only pushes ONE per slot — let's exercise with 1 slot and
    // verify the normal single-chunk path.
    auto f = std::async(std::launch::async, [&] { finalizeEmpty(ch); });

    int count = 0;
    ch->consume([&](StreamChunk&&) { count++; });
    f.wait();
    EXPECT_EQ(count, 1);
    EXPECT_TRUE(ch->isFinished());
}

TEST_F(StreamChannelTest, TryPopAfterFinalizerReturnsTerminalThenEmpty)
{
    auto ch = StreamChannel::create();
    finalizeEmpty(ch);
    auto c1 = ch->tryPop();
    ASSERT_TRUE(c1.has_value());
    EXPECT_TRUE(c1->finished);
    EXPECT_EQ(c1->reason, FinishReason::kError);
    EXPECT_FALSE(ch->tryPop().has_value());
}

TEST_F(StreamChannelTest, WaitPopUnblocksOnFinishWithEmptyQueue)
{
    auto ch = StreamChannel::create();
    std::promise<std::optional<StreamChunk>> p;
    auto fut = p.get_future();
    std::thread t([&] {
        // Consume the terminal chunk first.
        auto first = ch->waitPop(std::chrono::seconds{5});
        // Now the queue is empty but finished — this wait should return nullopt.
        auto second = ch->waitPop(std::chrono::milliseconds{500});
        p.set_value(second);
    });
    finalizeEmpty(ch);
    t.join();
    auto second = fut.get();
    EXPECT_FALSE(second.has_value());
    EXPECT_TRUE(ch->isFinished());
}

// ─────────────────────────────────────────────────────────────────────────────
// compactVector<SlotStreamState>
// ─────────────────────────────────────────────────────────────────────────────

class CompactVectorSlotStreamTest : public ::testing::Test
{
};

TEST_F(CompactVectorSlotStreamTest, EmptyVectorIsNoop)
{
    std::vector<SlotStreamState> v;
    std::vector<int32_t> mapping;
    rt::compactVector(mapping, v);
    EXPECT_TRUE(v.empty());
}

TEST_F(CompactVectorSlotStreamTest, NoEvictionPreservesOrderAndData)
{
    std::vector<SlotStreamState> v(3);
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i].sentTokenCount = 10 * (i + 1);
        v[i].pendingBytes = std::string(i + 1, 'x');
        v[i].terminalReason = static_cast<FinishReason>(i + 1);
    }
    std::vector<int32_t> mapping{0, 1, 2};
    rt::compactVector(mapping, v);
    ASSERT_EQ(v.size(), 3U);
    EXPECT_EQ(v[0].sentTokenCount, 10U);
    EXPECT_EQ(v[1].sentTokenCount, 20U);
    EXPECT_EQ(v[2].sentTokenCount, 30U);
    EXPECT_EQ(v[0].pendingBytes, "x");
    EXPECT_EQ(v[2].pendingBytes, "xxx");
}

TEST_F(CompactVectorSlotStreamTest, EvictMiddleKeepsEndpoints)
{
    std::vector<SlotStreamState> v(3);
    v[0].sentTokenCount = 100;
    v[1].sentTokenCount = 200;
    v[2].sentTokenCount = 300;
    std::vector<int32_t> mapping{0, -1, 1};
    rt::compactVector(mapping, v);
    ASSERT_EQ(v.size(), 2U);
    EXPECT_EQ(v[0].sentTokenCount, 100U);
    EXPECT_EQ(v[1].sentTokenCount, 300U);
}

TEST_F(CompactVectorSlotStreamTest, EvictAllYieldsEmpty)
{
    std::vector<SlotStreamState> v(2);
    v[0].sentTokenCount = 11;
    v[1].sentTokenCount = 22;
    std::vector<int32_t> mapping{-1, -1};
    rt::compactVector(mapping, v);
    EXPECT_TRUE(v.empty());
}

TEST_F(CompactVectorSlotStreamTest, ChannelSharedPtrSurvivesCompaction)
{
    auto ch0 = StreamChannel::create();
    auto ch1 = StreamChannel::create();
    auto ch2 = StreamChannel::create();
    std::vector<SlotStreamState> v(3);
    v[0].channel = ch0;
    v[1].channel = ch1;
    v[2].channel = ch2;
    std::vector<int32_t> mapping{-1, 0, 1};
    rt::compactVector(mapping, v);
    ASSERT_EQ(v.size(), 2U);
    EXPECT_EQ(v[0].channel.get(), ch1.get());
    EXPECT_EQ(v[1].channel.get(), ch2.get());
    // The evicted channel shared_ptr is released from the vector but still
    // alive via the caller's reference.
    EXPECT_EQ(ch0.use_count(), 1);
}

TEST_F(CompactVectorSlotStreamTest, SizeMismatchThrows)
{
    std::vector<SlotStreamState> v(2);
    std::vector<int32_t> mapping{0, 1, 2};
    EXPECT_THROW(rt::compactVector(mapping, v), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// LLMGenerationRequest.streamChannels layout check
// ─────────────────────────────────────────────────────────────────────────────

TEST(StreamingRequestTest, DefaultHasEmptyStreamChannels)
{
    rt::LLMGenerationRequest req;
    EXPECT_TRUE(req.streamChannels.empty());
}

TEST(StreamingRequestTest, CanAttachMixedChannels)
{
    rt::LLMGenerationRequest req;
    req.streamChannels.push_back(StreamChannel::create());
    req.streamChannels.push_back(nullptr);
    req.streamChannels.push_back(StreamChannel::create());
    EXPECT_EQ(req.streamChannels.size(), 3U);
    EXPECT_NE(req.streamChannels[0], nullptr);
    EXPECT_EQ(req.streamChannels[1], nullptr);
    EXPECT_NE(req.streamChannels[2], nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamChunk POD shape
// ─────────────────────────────────────────────────────────────────────────────

TEST(StreamChunkTest, DefaultConstructedIsEmptyNotFinished)
{
    StreamChunk c;
    EXPECT_TRUE(c.tokenIds.empty());
    EXPECT_TRUE(c.text.empty());
    EXPECT_FALSE(c.finished);
    EXPECT_EQ(c.reason, FinishReason::kNotFinished);
}

TEST(StreamChunkTest, IsMovable)
{
    StreamChunk c;
    c.tokenIds = {1, 2, 3};
    c.text = "abc";
    c.finished = true;
    c.reason = FinishReason::kEndId;
    StreamChunk d = std::move(c);
    EXPECT_EQ(d.tokenIds, (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(d.text, "abc");
    EXPECT_TRUE(d.finished);
    EXPECT_EQ(d.reason, FinishReason::kEndId);
}

// ─────────────────────────────────────────────────────────────────────────────
// Token callback emission
// ─────────────────────────────────────────────────────────────────────────────

TEST(TokenCallbackTest, EmitsAllNewTokensAndMarksOnlyLastFinished)
{
    rt::DecodingInferenceContext context;
    context.activeBatchSize = 1;
    context.generationRound = 7;
    context.tokenIds = {{10, 20, 30, 40}};
    context.finishedStates = {1};
    context.callbackEmittedTokenCounts = {2};

    std::vector<rt::TokenCallbackInfo> seen;
    context.onTokenGenerated = [&](rt::TokenCallbackInfo const& info) { seen.push_back(info); };

    rt::emitTokenCallbacks(context);

    ASSERT_EQ(seen.size(), 2U);
    EXPECT_EQ(seen[0].tokenId, 30);
    EXPECT_EQ(seen[0].batchIdx, 0);
    EXPECT_EQ(seen[0].generationStep, 7);
    EXPECT_FALSE(seen[0].isFinished);
    EXPECT_EQ(seen[1].tokenId, 40);
    EXPECT_EQ(seen[1].batchIdx, 0);
    EXPECT_EQ(seen[1].generationStep, 7);
    EXPECT_TRUE(seen[1].isFinished);
    EXPECT_EQ(context.callbackEmittedTokenCounts[0], 4);
}

TEST(TokenCallbackTest, DoesNotReplayPreviouslyEmittedTokens)
{
    rt::DecodingInferenceContext context;
    context.activeBatchSize = 1;
    context.generationRound = 3;
    context.tokenIds = {{11, 12}};
    context.finishedStates = {0};
    context.callbackEmittedTokenCounts = {0};

    std::vector<rt::TokenCallbackInfo> seen;
    context.onTokenGenerated = [&](rt::TokenCallbackInfo const& info) { seen.push_back(info); };

    rt::emitTokenCallbacks(context);
    rt::emitTokenCallbacks(context);

    ASSERT_EQ(seen.size(), 2U);
    EXPECT_EQ(seen[0].tokenId, 11);
    EXPECT_EQ(seen[1].tokenId, 12);
    EXPECT_EQ(context.callbackEmittedTokenCounts[0], 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// applyStopStringMatch — pure helper for hold-back stream matching
// ─────────────────────────────────────────────────────────────────────────────

TEST(ApplyStopStringMatchTest, NoMatchFinalFlushes)
{
    std::string buffer = "hello##";
    auto out = rt::applyStopStringMatch(buffer, {"###"}, /*maxStopLen=*/3, /*isFinal=*/true);
    EXPECT_EQ(out.emitted, "hello##");
    EXPECT_FALSE(out.stopMatched);
    EXPECT_TRUE(buffer.empty());
}

TEST(ApplyStopStringMatchTest, CrossChunkMatch)
{
    // Simulate two iterations: first appends "abc" (held), second appends "de###".
    std::string buffer;
    buffer.append("abc");
    auto out1 = rt::applyStopStringMatch(buffer, {"###"}, /*maxStopLen=*/3, /*isFinal=*/false);
    EXPECT_EQ(out1.emitted, "a"); // S-1=2 held → emit "a", retain "bc"
    EXPECT_FALSE(out1.stopMatched);
    EXPECT_EQ(buffer, "bc");

    buffer.append("de###");
    auto out2 = rt::applyStopStringMatch(buffer, {"###"}, /*maxStopLen=*/3, /*isFinal=*/false);
    EXPECT_EQ(out2.emitted, "bcde");
    EXPECT_TRUE(out2.stopMatched);
    EXPECT_TRUE(buffer.empty());
}

TEST(ApplyStopStringMatchTest, EarliestPositionWins)
{
    // "END" at pos 9, "###" at pos 3 ⇒ "###" wins by position, not list order.
    std::string buffer = "foo###barEND";
    auto out = rt::applyStopStringMatch(buffer, {"END", "###"}, /*maxStopLen=*/3, /*isFinal=*/false);
    EXPECT_EQ(out.emitted, "foo");
    EXPECT_TRUE(out.stopMatched);
    EXPECT_TRUE(buffer.empty());
}

TEST(ApplyStopStringMatchTest, HoldBackThenFinalFlush)
{
    // Held bytes from a no-match iteration must be flushed when isFinal fires
    // without a match (e.g. stream ends with the model holding chars that
    // *could* have started a stop). Otherwise the trailing characters are lost.
    std::string buffer = "ab";
    auto out1 = rt::applyStopStringMatch(buffer, {"###"}, /*maxStopLen=*/3, /*isFinal=*/false);
    EXPECT_TRUE(out1.emitted.empty()); // S-1=2 ≥ buffer.size(), nothing safe to emit
    EXPECT_EQ(buffer, "ab");

    auto out2 = rt::applyStopStringMatch(buffer, {"###"}, /*maxStopLen=*/3, /*isFinal=*/true);
    EXPECT_EQ(out2.emitted, "ab");
    EXPECT_FALSE(out2.stopMatched);
    EXPECT_TRUE(buffer.empty());
}

TEST(ApplyStopStringMatchTest, MultiByteUtf8StopMatches)
{
    // "你好" is E4 BD A0 E5 A5 BD (6 bytes). Algorithm is byte-level — verify
    // it finds the stop correctly regardless of multi-byte codepoints.
    std::string const stopCh = "\xE4\xBD\xA0\xE5\xA5\xBD"; // "你好"
    std::string buffer = "hello" + stopCh + "tail";
    auto out = rt::applyStopStringMatch(buffer, {stopCh}, /*maxStopLen=*/stopCh.size(), /*isFinal=*/false);
    EXPECT_EQ(out.emitted, "hello");
    EXPECT_TRUE(out.stopMatched);
    EXPECT_TRUE(buffer.empty());
}
