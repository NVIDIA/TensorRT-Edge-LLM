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
// Interactive streaming demo.
//
// Reads an llm_inference-compatible JSON input file (supports text / image /
// audio content types via examples/utils/requestFileParser.h), then processes
// each request one-by-one, streaming the response to stdout chunk-by-chunk.
//
// While a response is streaming the user can press:
//   's'       — skip the current request (channel->cancel())
//   'q'       — quit (cancel current + stop iterating)
//   Ctrl-C    — same as 'q'
//
// The terminal is temporarily switched to raw mode (no line buffering, no echo)
// so single keystrokes are picked up immediately. An RAII guard restores the
// original termios on every exit path (normal, signal, exception).
//
// Usage:
//   llm_stream --engineDir DIR --inputFile PATH
//     [--multimodalEngineDir DIR] [--specDecode [--specDraftTopK K ...]]
//     [--maxGenerateLength N] [--streamInterval N]

#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "requestFileParser.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <poll.h>
#include <termios.h>
#include <unistd.h>

using namespace trt_edgellm;

namespace
{

// ── Arg parsing ─────────────────────────────────────────────────────────────

struct Args
{
    std::string engineDir;
    std::string multimodalEngineDir;
    std::string inputFile;
    int32_t maxGenerateLength = -1; // Uses value from JSON if < 0.
    int32_t streamInterval = 1;
    bool showSpecialTokens = false; // StreamChannel::setSkipSpecialTokens(!this)
    bool specDecode = false;
    int32_t specDraftTopK = 8;
    bool specDraftTopKSet = false;
    int32_t specDraftStep = 4;
    bool specDraftStepSet = false;
    int32_t specVerifySize = 24;
    bool specVerifySizeSet = false;
    int32_t dflashBlockSize = 0;
};

namespace
{

std::filesystem::path getBaseConfigPath(std::string const& engineDir)
{
    std::filesystem::path const dir{engineDir};
    std::filesystem::path const configPath = dir / "config.json";
    if (std::filesystem::is_regular_file(configPath))
    {
        return configPath;
    }
    return dir / "base_config.json";
}

std::filesystem::path getDraftConfigPath(std::string const& engineDir)
{
    return std::filesystem::path{engineDir} / "draft_config.json";
}

int32_t maxVerifySizeOrDefault(rt::LLMEngineConfig const& config, int32_t fallback)
{
    return config.maxVerifyTreeSize > 0 ? config.maxVerifyTreeSize : fallback;
}

int32_t dsparkVerifySizeOrDefault(std::string const& engineDir)
{
    std::filesystem::path const draftConfigPath = getDraftConfigPath(engineDir);
    if (!std::filesystem::is_regular_file(draftConfigPath))
    {
        return 8;
    }

    rt::LLMEngineConfig const draftConfig = rt::parseDraftEngineConfig(draftConfigPath);
    return draftConfig.specDraftBlockSize > 0 ? draftConfig.specDraftBlockSize + 1 : 8;
}

int32_t cachedBlockDraftBlockSizeOrThrow(
    std::string const& engineDir, rt::LLMEngineConfig const& baseConfig, int32_t explicitBlockSize)
{
    if (explicitBlockSize > 0)
    {
        return explicitBlockSize;
    }

    std::filesystem::path const draftConfigPath = getDraftConfigPath(engineDir);
    if (std::filesystem::is_regular_file(draftConfigPath))
    {
        rt::LLMEngineConfig const draftConfig = rt::parseDraftEngineConfig(draftConfigPath);
        if (draftConfig.specDraftBlockSize > 0)
        {
            return draftConfig.specDraftBlockSize;
        }
    }
    if (baseConfig.specDraftBlockSize > 0)
    {
        return baseConfig.specDraftBlockSize;
    }

    throw std::runtime_error(
        "unable to resolve DFlash/JetSpec block size from CLI, draft_config.json, or base config.");
}

bool applyEngineSpecDecodeDefaults(Args& args)
{
    if (!args.specDecode)
    {
        return true;
    }

    try
    {
        rt::LLMEngineConfig const baseConfig = rt::parseEngineConfig(getBaseConfigPath(args.engineDir));
        switch (baseConfig.specDecodeType)
        {
        case rt::SpecDecodeMode::kDFlash:
        case rt::SpecDecodeMode::kJetSpec:
        {
            if (!args.specDraftTopKSet)
            {
                args.specDraftTopK = 1;
            }
            if (!args.specDraftStepSet)
            {
                args.specDraftStep = 1;
            }
            int32_t const blockSize
                = cachedBlockDraftBlockSizeOrThrow(args.engineDir, baseConfig, args.dflashBlockSize);
            if (args.dflashBlockSize == 0)
            {
                args.dflashBlockSize = blockSize;
            }
            if (!args.specVerifySizeSet)
            {
                args.specVerifySize = args.specDraftTopK > 1 ? maxVerifySizeOrDefault(baseConfig, 128) : blockSize;
            }
            break;
        }
        case rt::SpecDecodeMode::kDSpark:
            if (!args.specDraftTopKSet)
            {
                args.specDraftTopK = 1;
            }
            if (!args.specDraftStepSet)
            {
                args.specDraftStep = 1;
            }
            if (!args.specVerifySizeSet)
            {
                args.specVerifySize = dsparkVerifySizeOrDefault(args.engineDir);
            }
            break;
        default: break;
        }

        bool const isCachedBlockDraft = baseConfig.specDecodeType == rt::SpecDecodeMode::kDFlash
            || baseConfig.specDecodeType == rt::SpecDecodeMode::kJetSpec;
        std::cout << "Spec decode engine mode: " << rt::specDecodeModeName(baseConfig.specDecodeType) << "\n"
                  << "Resolved spec defaults: topK=" << args.specDraftTopK << ", step=" << args.specDraftStep
                  << ", verifySize=" << args.specVerifySize;
        if (isCachedBlockDraft)
        {
            std::cout << ", blockSize=" << args.dflashBlockSize;
        }
        std::cout << "\n";
    }
    catch (std::exception const& e)
    {
        std::cerr << "Failed to resolve speculative decoding defaults from engine config: " << e.what() << "\n";
        return false;
    }

    return true;
}

} // namespace

void printUsage(char const* argv0)
{
    std::cout << "Usage: " << argv0
              << " --engineDir DIR --inputFile PATH\n"
                 "           [--multimodalEngineDir DIR] [--maxGenerateLength N]\n"
                 "           [--streamInterval N] [--specDecode [--specDraftTopK K]\n"
                 "                                             [--specDraftStep S]\n"
                 "                                             [--specVerifySize V]\n"
                 "                                             [--dflashBlockSize B|--jetspecBlockSize B]]\n\n"
                 "Spec decode:\n"
                 "   --specDraftTopK K     Default: 8; DFlash/JetSpec/DSpark default to 1 when omitted\n"
                 "   --specDraftStep S     Default: 4; DFlash/JetSpec/DSpark default to 1 when omitted\n"
                 "   --specVerifySize V    Default: 24; DFlash/JetSpec linear default to block size\n"
                 "   --dflashBlockSize B   DFlash/JetSpec proposal block size; 0 means infer from engine config\n\n"
                 "Hotkeys while streaming:\n"
                 "   s         skip the current request\n"
                 "   q         quit (cancel current + stop)\n"
                 "   Ctrl-C    same as q\n";
}

bool parseArgs(int argc, char** argv, Args& args)
{
    auto takeStr = [&](int& i, std::string const& name, std::string& out) -> bool {
        if (i + 1 >= argc)
        {
            std::cerr << "Missing value for " << name << "\n";
            return false;
        }
        out = argv[++i];
        return true;
    };
    auto takeInt = [&](int& i, std::string const& name, int32_t& out) -> bool {
        std::string s;
        if (!takeStr(i, name, s))
        {
            return false;
        }
        try
        {
            out = std::stoi(s);
        }
        catch (std::exception const&)
        {
            std::cerr << "Invalid integer for " << name << ": " << s << "\n";
            return false;
        }
        return true;
    };

    for (int i = 1; i < argc; ++i)
    {
        std::string const a = argv[i];
        if (a == "--engineDir")
        {
            if (!takeStr(i, a, args.engineDir))
            {
                return false;
            }
        }
        else if (a == "--multimodalEngineDir")
        {
            if (!takeStr(i, a, args.multimodalEngineDir))
            {
                return false;
            }
        }
        else if (a == "--inputFile")
        {
            if (!takeStr(i, a, args.inputFile))
            {
                return false;
            }
        }
        else if (a == "--maxGenerateLength")
        {
            if (!takeInt(i, a, args.maxGenerateLength))
            {
                return false;
            }
        }
        else if (a == "--streamInterval")
        {
            if (!takeInt(i, a, args.streamInterval))
            {
                return false;
            }
        }
        else if (a == "--showSpecialTokens")
        {
            args.showSpecialTokens = true;
        }
        else if (a == "--specDecode" || a == "--eagle")
        {
            args.specDecode = true;
        }
        else if (a == "--specDraftTopK" || a == "--eagleDraftTopK")
        {
            if (!takeInt(i, a, args.specDraftTopK))
            {
                return false;
            }
            args.specDraftTopKSet = true;
        }
        else if (a == "--specDraftStep" || a == "--eagleDraftStep")
        {
            if (!takeInt(i, a, args.specDraftStep))
            {
                return false;
            }
            args.specDraftStepSet = true;
        }
        else if (a == "--specVerifySize" || a == "--specVerifyTreeSize" || a == "--eagleVerifyTreeSize")
        {
            if (!takeInt(i, a, args.specVerifySize))
            {
                return false;
            }
            args.specVerifySizeSet = true;
            if (args.specVerifySize < 0)
            {
                std::cerr << "--specVerifySize must be non-negative\n";
                return false;
            }
        }
        else if (a == "--dflashBlockSize" || a == "--jetspecBlockSize")
        {
            if (!takeInt(i, a, args.dflashBlockSize))
            {
                return false;
            }
            if (args.dflashBlockSize < 0)
            {
                std::cerr << "--dflashBlockSize/--jetspecBlockSize must be non-negative\n";
                return false;
            }
        }
        else if (a == "-h" || a == "--help")
        {
            printUsage(argv[0]);
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown arg: " << a << "\n";
            return false;
        }
    }
    if (args.engineDir.empty() || args.inputFile.empty())
    {
        std::cerr << "Both --engineDir and --inputFile are required.\n";
        return false;
    }
    return true;
}

// ── Terminal raw mode + hotkey listener ─────────────────────────────────────

//! RAII: flip stdin to raw mode on construction, restore original termios on
//! destruction. Only activates when stdin is a TTY.
class RawTerminal
{
public:
    RawTerminal() noexcept
    {
        mIsTty = isatty(STDIN_FILENO) != 0;
        if (!mIsTty)
        {
            return;
        }
        if (tcgetattr(STDIN_FILENO, &mOriginal) != 0)
        {
            mIsTty = false;
            return;
        }
        termios raw = mOriginal;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0)
        {
            mIsTty = false;
            return;
        }
        mActive = true;
    }

    ~RawTerminal() noexcept
    {
        restore();
    }

    RawTerminal(RawTerminal const&) = delete;
    RawTerminal& operator=(RawTerminal const&) = delete;

    bool active() const noexcept
    {
        return mActive;
    }

    void restore() noexcept
    {
        if (mActive)
        {
            tcsetattr(STDIN_FILENO, TCSANOW, &mOriginal);
            mActive = false;
        }
    }

private:
    termios mOriginal{};
    bool mIsTty{false};
    bool mActive{false};
};

// User-action flags set by the input watcher or the signal handler.
std::atomic<bool> gSkipCurrent{false};
std::atomic<bool> gQuitRequested{false};

void sigintHandler(int /*sig*/)
{
    gQuitRequested.store(true, std::memory_order_release);
}

//! Background thread: poll stdin in raw mode. When the user presses 's' flip
//! gSkipCurrent, when they press 'q' flip gQuitRequested. Exits when
//! `stopSignal` becomes true.
void inputWatcherLoop(std::atomic<bool> const& stopSignal)
{
    while (!stopSignal.load(std::memory_order_acquire))
    {
        pollfd pfd{STDIN_FILENO, POLLIN, 0};
        int rv = poll(&pfd, 1, 50 /* ms */);
        if (rv <= 0)
        {
            continue;
        }
        if (!(pfd.revents & POLLIN))
        {
            continue;
        }
        char buf[16] = {0};
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        for (ssize_t i = 0; i < n; ++i)
        {
            switch (buf[i])
            {
            case 's':
            case 'S': gSkipCurrent.store(true, std::memory_order_release); break;
            case 'q':
            case 'Q':
            case 0x03 /* Ctrl-C fallback, normally captured by SIGINT */:
                gQuitRequested.store(true, std::memory_order_release);
                break;
            default: break;
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

//! Extract a short preview of the prompt (last user message).
std::string promptPreview(rt::LLMGenerationRequest::Request const& req, size_t maxChars = 80)
{
    for (auto it = req.messages.rbegin(); it != req.messages.rend(); ++it)
    {
        if (it->role != "user")
        {
            continue;
        }
        for (auto const& c : it->contents)
        {
            if (c.type == "text" && !c.content.empty())
            {
                if (c.content.size() <= maxChars)
                {
                    return c.content;
                }
                return c.content.substr(0, maxChars) + "...";
            }
        }
    }
    return "<no text content>";
}

//! Count multimodal buffers on a single request for the banner line.
std::string multimodalTag(rt::LLMGenerationRequest::Request const& req)
{
    std::string out;
    if (!req.imageBuffers.empty())
    {
        out += " [" + std::to_string(req.imageBuffers.size()) + " image]";
    }
    if (!req.audioBuffers.empty())
    {
        out += " [" + std::to_string(req.audioBuffers.size()) + " audio]";
    }
    return out;
}

// ── Per-request streaming ───────────────────────────────────────────────────

struct RequestResult
{
    bool workerOk{false};
    bool cancelledBySkip{false};
    bool quitMidGeneration{false};
    rt::FinishReason finishReason{rt::FinishReason::kNotFinished};
    size_t tokensOut{0};
    double ttftMs{0.0};
    double totalMs{0.0};
};

RequestResult runOneRequest(rt::LLMInferenceRuntime& runtime, cudaStream_t stream,
    rt::LLMGenerationRequest batchRequest, int32_t streamInterval, bool showSpecialTokens)
{
    RequestResult result;

    auto ch = rt::StreamChannel::create();
    ch->setStreamInterval(streamInterval);
    ch->setSkipSpecialTokens(!showSpecialTokens);
    batchRequest.streamChannels.push_back(ch);

    rt::LLMGenerationResponse response;
    std::atomic<bool> workerOk{false};

    auto const t0 = std::chrono::steady_clock::now();
    std::thread worker([&] { workerOk = runtime.handleRequest(batchRequest, response, stream); });

    // Reset per-request flags so earlier skip/quit state doesn't leak.
    gSkipCurrent.store(false, std::memory_order_release);

    bool firstChunk = true;
    ch->consume([&](rt::StreamChunk&& c) {
        if (firstChunk)
        {
            auto const now = std::chrono::steady_clock::now();
            result.ttftMs = std::chrono::duration<double, std::milli>(now - t0).count();
            firstChunk = false;
        }
        result.tokensOut += c.tokenIds.size();

        // Flush per chunk so the user sees text appear live.
        if (!c.text.empty())
        {
            std::fwrite(c.text.data(), 1, c.text.size(), stdout);
            std::fflush(stdout);
        }

        // When logprobs were requested (num_logprobs > 0), print each new token's
        // top-K to stderr — kept off stdout so it does not interleave with the live
        // text stream. Each entry is (token_id, log-prob).
        for (auto const& stepEntries : c.logprobs)
        {
            std::fprintf(stderr, "[logprobs]");
            for (auto const& e : stepEntries)
            {
                std::fprintf(stderr, " (%d, %.4f)", e.tokenId, e.logprob);
            }
            std::fprintf(stderr, "\n");
        }
        if (c.finished)
        {
            result.finishReason = c.reason;
        }

        // Poll user intents AFTER delivering the chunk — never drop output.
        if (gQuitRequested.load(std::memory_order_acquire))
        {
            result.quitMidGeneration = true;
            ch->cancel();
        }
        else if (gSkipCurrent.load(std::memory_order_acquire))
        {
            result.cancelledBySkip = true;
            ch->cancel();
        }
    });
    worker.join();

    auto const t1 = std::chrono::steady_clock::now();
    result.totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.workerOk = workerOk.load();
    // Prefer channel-latched reason if the consume loop exited before the runtime's finish.
    if (result.finishReason == rt::FinishReason::kNotFinished && ch->isFinished())
    {
        result.finishReason = ch->getReason();
    }
    return result;
}

void printRequestFooter(RequestResult const& r)
{
    std::printf("\n  ↳ %zu tokens, TTFT %.1fms, total %.1fms (%.1f tok/s), finish=%s%s\n", r.tokensOut, r.ttftMs,
        r.totalMs, r.totalMs > 0 ? static_cast<double>(r.tokensOut) * 1000.0 / r.totalMs : 0.0,
        rt::finishReasonName(r.finishReason),
        r.cancelledBySkip ? " [SKIPPED]" : (r.quitMidGeneration ? " [QUIT]" : ""));
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(argc, argv, args))
    {
        printUsage(argv[0]);
        return 1;
    }
    if (!std::filesystem::is_directory(args.engineDir))
    {
        std::cerr << "Engine dir not found: " << args.engineDir << "\n";
        return 1;
    }
    if (!std::filesystem::is_regular_file(args.inputFile))
    {
        std::cerr << "Input file not found: " << args.inputFile << "\n";
        return 1;
    }
    if (!applyEngineSpecDecodeDefaults(args))
    {
        return 1;
    }

    auto pluginHandles = loadEdgellmPluginLib();

    // Parse the input JSON into a flat list of per-request batches-of-one —
    // streaming is per-request so each request is its own LLMGenerationRequest.
    std::unordered_map<std::string, std::string> loraMap;
    std::vector<rt::LLMGenerationRequest::Request> singleRequests;
    rt::LLMGenerationRequest requestTemplate; // sampling / maxGen fields copied from the file
    bool templateInitialized = false;

    try
    {
        auto [lm, batches]
            = exampleUtils::parseRequestFile(args.inputFile, /*batchSizeOverride=*/1, args.maxGenerateLength);
        loraMap = std::move(lm);
        for (auto& b : batches)
        {
            if (!templateInitialized)
            {
                requestTemplate = b; // keep sampling params
                requestTemplate.requests.clear();
                requestTemplate.streamChannels.clear();
                requestTemplate.formattedRequests.clear();
                templateInitialized = true;
            }
            for (auto& r : b.requests)
            {
                singleRequests.push_back(std::move(r));
            }
        }
    }
    catch (std::exception const& e)
    {
        std::cerr << "Failed to parse input file: " << e.what() << "\n";
        return 2;
    }

    if (singleRequests.empty())
    {
        std::cerr << "No requests in input file.\n";
        return 2;
    }
    std::cout << "Loaded " << singleRequests.size() << " request(s) from " << args.inputFile << "\n";

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    try
    {
        if (args.specDecode)
        {
            rt::SpecDecodeDraftingConfig draft{};
            draft.draftingTopK = args.specDraftTopK;
            draft.draftingStep = args.specDraftStep;
            draft.verifySize = args.specVerifySize;
            draft.dflashBlockSize = args.dflashBlockSize;
            runtime = std::make_unique<rt::LLMInferenceRuntime>(
                args.engineDir, args.multimodalEngineDir, loraMap, draft, stream);
        }
        else
        {
            runtime
                = std::make_unique<rt::LLMInferenceRuntime>(args.engineDir, args.multimodalEngineDir, loraMap, stream);
        }
    }
    catch (std::exception const& e)
    {
        std::cerr << "Runtime init failed: " << e.what() << "\n";
        return 3;
    }

    runtime->captureDecodingCUDAGraph(stream);

    std::signal(SIGINT, sigintHandler);

    // Raw mode + watcher thread — scoped so we always restore termios, even on
    // exception paths out of the generation loop.
    RawTerminal raw;
    std::atomic<bool> watcherStop{false};
    std::thread watcher;
    if (raw.active())
    {
        watcher = std::thread(inputWatcherLoop, std::ref(watcherStop));
    }

    std::cout << "\nHotkeys: [s]=skip  [q]=quit  Ctrl-C=quit\n";
    std::cout << "========================================\n";

    size_t const total = singleRequests.size();
    int rc = 0;
    for (size_t i = 0; i < total; ++i)
    {
        if (gQuitRequested.load(std::memory_order_acquire))
        {
            std::cout << "\n[quit] Stopping before request " << (i + 1) << "/" << total << "\n";
            break;
        }

        std::printf("\n[%zu/%zu] %s%s\n> ", i + 1, total, promptPreview(singleRequests[i]).c_str(),
            multimodalTag(singleRequests[i]).c_str());
        std::fflush(stdout);

        rt::LLMGenerationRequest req = requestTemplate;
        req.requests.push_back(std::move(singleRequests[i]));

        RequestResult r;
        try
        {
            r = runOneRequest(*runtime, stream, std::move(req), args.streamInterval, args.showSpecialTokens);
        }
        catch (std::exception const& e)
        {
            std::cerr << "\nRequest " << (i + 1) << " failed: " << e.what() << "\n";
            rc = 4;
            continue;
        }
        if (!r.workerOk)
        {
            std::cerr << "\nRequest " << (i + 1) << " returned failure\n";
            rc = 4;
        }
        printRequestFooter(r);

        if (r.quitMidGeneration)
        {
            std::cout << "\n[quit] Stopping after this request.\n";
            break;
        }
    }

    // Shut down the watcher before restoring termios (otherwise read() can race).
    watcherStop.store(true, std::memory_order_release);
    if (watcher.joinable())
    {
        watcher.join();
    }
    raw.restore();

    runtime.reset();
    cudaStreamDestroy(stream);
    std::cout << "\n========================================\nDone.\n";
    return rc;
}
