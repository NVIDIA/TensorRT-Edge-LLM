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

#include <cstring>

inline constexpr int no_argument{0};
inline constexpr int required_argument{1};
inline constexpr int optional_argument{2};

struct option
{
    char const* name;
    int has_arg;
    int* flag;
    int val;
};

inline char* optarg{nullptr};
inline int optind{1};
inline int opterr{1};
inline int optopt{0};
inline char* optcursor{nullptr};

inline int getopt_long(
    int argc, char* const argv[], char const* shortOptions, option const* longOptions, int* longIndex)
{
    optarg = nullptr;
    if (optcursor == nullptr || *optcursor == '\0')
    {
        optcursor = nullptr;
        if (optind >= argc || argv[optind][0] != '-' || argv[optind][1] == '\0')
        {
            return -1;
        }

        if (std::strcmp(argv[optind], "--") == 0)
        {
            ++optind;
            return -1;
        }

        if (argv[optind][1] != '-')
        {
            optcursor = argv[optind++] + 1;
        }
    }

    if (optcursor != nullptr)
    {
        int const result = *optcursor++;
        char const* optionSpec = std::strchr(shortOptions, result);
        if (optionSpec == nullptr || result == ':')
        {
            optopt = result;
            return '?';
        }

        if (optionSpec[1] == ':')
        {
            bool const optionalArgument = optionSpec[2] == ':';
            if (*optcursor != '\0')
            {
                optarg = optcursor;
                optcursor = nullptr;
            }
            else if (!optionalArgument && optind < argc)
            {
                optarg = argv[optind++];
                optcursor = nullptr;
            }
            else if (!optionalArgument)
            {
                optopt = result;
                optcursor = nullptr;
                return shortOptions[0] == ':' ? ':' : '?';
            }
        }
        else if (*optcursor == '\0')
        {
            optcursor = nullptr;
        }
        return result;
    }

    char* argument = argv[optind++] + 2;
    char* value = std::strchr(argument, '=');
    if (value != nullptr)
    {
        *value++ = '\0';
    }
    for (int index = 0; longOptions[index].name != nullptr; ++index)
    {
        option const& current = longOptions[index];
        if (std::strcmp(argument, current.name) != 0)
        {
            continue;
        }
        if (longIndex != nullptr)
        {
            *longIndex = index;
        }
        if (current.has_arg == required_argument && value == nullptr)
        {
            if (optind >= argc)
            {
                optopt = current.val;
                return '?';
            }
            value = argv[optind++];
        }
        else if (current.has_arg == no_argument && value != nullptr)
        {
            optopt = current.val;
            return '?';
        }
        optarg = value;
        if (current.flag != nullptr)
        {
            *current.flag = current.val;
            return 0;
        }
        return current.val;
    }

    return '?';
}
