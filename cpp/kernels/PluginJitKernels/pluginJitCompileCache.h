/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <exception>
#include <future>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace trt_edgellm
{

//! Share in-flight and completed plugin JIT compilations by semantic key.
template <typename Key, typename Result, typename Hasher>
class PluginJitCompileCache
{
public:
    template <typename CompileFunction>
    Result getOrCompile(Key const& key, CompileFunction&& compile)
    {
        std::shared_future<Result> future;
        std::promise<Result> promise;
        bool compileThisThread{false};
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto const findIter = mCache.find(key);
            if (findIter != mCache.end())
            {
                future = findIter->second;
            }
            else
            {
                compileThisThread = true;
                future = promise.get_future().share();
                mCache.emplace(key, future);
            }
        }

        if (!compileThisThread)
        {
            return future.get();
        }

        try
        {
            Result result = std::forward<CompileFunction>(compile)();
            promise.set_value(std::move(result));
        }
        catch (...)
        {
            std::exception_ptr const exception = std::current_exception();
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mCache.erase(key);
            }
            promise.set_exception(exception);
            std::rethrow_exception(exception);
        }
        return future.get();
    }

private:
    std::mutex mMutex;
    std::unordered_map<Key, std::shared_future<Result>, Hasher> mCache;
};

} // namespace trt_edgellm
