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

#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
#include <chrono>
#include <nvdvms_client.h>
#include <thread>
#endif

namespace trt_edgellm::safety::engine_test
{

//! Test-only phase observer. An external owner coordinates the whole VM; this class never changes its state.
class TestLifecycle
{
public:
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    static constexpr bool kQNX_SAFETY = true;
#else
    static constexpr bool kQNX_SAFETY = false;
#endif

    //! Require initialization before any explicit CUDA calls.
    bool initialize()
    {
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
        if (nvdvms_init() != NvDvmsSuccess)
        {
            std::fprintf(stderr, "nvdvms_init failed; verify ClientCommGetState ability.\n");
            return false;
        }
        return wait(NVDVMS_INIT_DONE, NVDVMS_INIT_DONE);
#else
        return true;
#endif
    }

    //! Called after all allocations, module preparation, bindings and warmup have completed.
    bool operational()
    {
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
        return wait(NVDVMS_INIT_DONE, NVDVMS_OPERATIONAL);
#else
        return true;
#endif
    }

    //! Called only after healthy, synchronized execution; CUDA teardown must not precede this gate.
    bool deinitialize()
    {
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
        return wait(NVDVMS_OPERATIONAL, NVDVMS_DEINIT_PREPARE);
#else
        return true;
#endif
    }

    //! Release the observer after CUDA teardown; the VM owner must shut down/reboot, not reinitialize CUDA.
    bool finish()
    {
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
        return nvdvms_deinit() == NvDvmsSuccess;
#else
        return true;
#endif
    }

private:
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    bool wait(NvDvmsState previous, NvDvmsState expected)
    {
        std::printf(
            "LIFECYCLE_READY previous=%d expected=%d\n", static_cast<int>(previous), static_cast<int>(expected));
        std::fflush(stdout);
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        do
        {
            NvDvmsState state{};
            auto const status = nvdvms_get_vm_state(&state);
            if (status != NvDvmsSuccess)
            {
                std::fprintf(stderr, "nvdvms_get_vm_state failed: %d\n", static_cast<int>(status));
                return false;
            }
            if (state == expected)
            {
                std::printf("LIFECYCLE_OBSERVED state=%d\n", static_cast<int>(state));
                std::fflush(stdout);
                return true;
            }
            bool const enteringOperational = expected == NVDVMS_OPERATIONAL && state == NVDVMS_OPERATIONAL_START;
            if (state != previous && !enteringOperational)
            {
                std::fprintf(stderr, "Unexpected VM state %d; expected %d.\n", static_cast<int>(state),
                    static_cast<int>(expected));
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } while (std::chrono::steady_clock::now() < deadline);
        std::fprintf(stderr, "Timed out waiting for external VM lifecycle coordinator.\n");
        return false;
    }
#endif
};

//! Avoid implicit CUDA library teardown outside the controlled Deinit Preparation path.
inline int finishTestProcess(int status)
{
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    std::fflush(nullptr);
    std::_Exit(status);
#else
    return status;
#endif
}

} // namespace trt_edgellm::safety::engine_test
