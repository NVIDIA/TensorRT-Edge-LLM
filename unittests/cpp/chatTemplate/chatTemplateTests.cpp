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

#include "chatTemplate/chatTemplate.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace trt_edgellm;

namespace
{
class TemporaryTemplate
{
public:
    TemporaryTemplate(std::initializer_list<std::pair<std::filesystem::path, std::string>> files)
        : path(std::filesystem::temp_directory_path() / "edgellm_chat_template_test")
    {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
        for (auto const& [filename, source] : files)
        {
            std::filesystem::create_directories((path / filename).parent_path());
            std::ofstream(path / filename) << source;
        }
    }

    ~TemporaryTemplate()
    {
        std::filesystem::remove_all(path);
    }

    std::filesystem::path path;
};
} // namespace

TEST(ChatTemplateTest, AppliesProviderJinjaToStructuredConversation)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- for message in messages -%}{{- message.role -}}:{%- if isArray(message.content) -%}{%- for item in message.content -%}{%- if item.type == "image" -%}<image>{%- else -%}{{ item.text }}{%- endif -%}{%- endfor -%}{%- else if isString(message.content) -%}{{ message.content }}{%- endif -%}{%- for call in message.tool_calls -%}<call name="{{ call.function.name }}">{{ call.function.arguments | tojson }}</call>{%- endfor -%}{{ "\n" }}{%- endfor -%}{%- if length(tools) > 0 -%}<tools>{{ tools | tojson }}</tools>{%- endif -%}{%- if add_generation_prompt -%}assistant:{%- endif -%})JINJA"}};

    rt::Message user{"user", {{"image", ""}, {"text", "describe"}}};
    user.contentIsArray = true;
    rt::Message assistant{"assistant", {}};
    assistant.contentIsNull = true;
    assistant.toolCalls.push_back({"call_1", "function", "inspect", R"({"detail":"high"})", false});

    rt::LLMGenerationRequest::Request request;
    request.messages = {std::move(user), std::move(assistant)};
    chat_template::ChatTemplate::Options options;
    options.tools.push_back({"inspect", "Inspect an image", R"({"type":"object"})", false});

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(request, formatted, options));
    EXPECT_EQ(formatted.formattedCompleteRequest,
        "user:<image>describe\nassistant:<call name=\"inspect\">{\"detail\": \"high\"}</call>\n"
        "<tools>[{\"type\": \"function\", \"function\": {\"name\": \"inspect\", \"description\": \"Inspect an "
        "image\", \"parameters\": {\"type\": \"object\"}}}]</tools>assistant:");
}

TEST(ChatTemplateTest, LoadsProviderTemplateDirectly)
{
    TemporaryTemplate model{{"chat_template.jinja", "{{ messages.2.content }}"}};

    rt::LLMGenerationRequest::Request request;
    request.messages = {rt::Message{"system", {{"text", "system"}}}, rt::Message{"user", {{"text", " first "}}},
        rt::Message{"assistant", {{"text", " answer "}}}, rt::Message{"user", {{"text", " latest "}}}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(request, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, " answer ");
}

TEST(ChatTemplateTest, RejectsNamedProviderTemplates)
{
    TemporaryTemplate model{
        {"chat_template.jinja", "default"}, {"additional_chat_templates/tool_use.jinja", "tool-use"}};

    chat_template::ChatTemplate renderer;
    EXPECT_FALSE(renderer.load(model.path));
}

TEST(ChatTemplateTest, RejectsLegacyJsonTemplateArtifacts)
{
    TemporaryTemplate model{
        {"chat_template.json", R"({"chat_template":"ignored"})"}, {"processed_chat_template.json", R"({"roles":{}})"}};

    chat_template::ChatTemplate renderer;
    EXPECT_FALSE(renderer.load(model.path));
}

TEST(ChatTemplateTest, RejectsProviderAndNativeRendererTogether)
{
    TemporaryTemplate model{
        {"chat_template.jinja", "{{ messages[0].content }}"}, {"chat_template.model", "qwen3_tts\n"}};

    chat_template::ChatTemplate renderer;
    EXPECT_FALSE(renderer.load(model.path));
}

TEST(ChatTemplateTest, NormalizesContentToProviderTemplateContract)
{
    TemporaryTemplate structured{{"chat_template.jinja",
        "{% for item in messages[0].content %}{{ item.text }}{% if not loop.last %}|{% endif %}{% endfor %}"}};
    rt::Message text{"user", {{"text", "plain string"}}};
    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(structured.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{text}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "plain string");

    TemporaryTemplate stringOnly{{"chat_template.jinja", "{{ messages[0].content }}"}};
    text.contents = {{"text", ""}, {"text", "second"}};
    text.contentIsArray = true;
    ASSERT_TRUE(renderer.load(stringOnly.path));
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{text}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "\nsecond");

    TemporaryTemplate filtered{{"chat_template.jinja",
        "{% for message in messages if message.content is not none %}{{ message.content }}{% endfor %}"}};
    text.contents = {{"text", "plain string"}};
    text.contentIsArray = false;
    ASSERT_TRUE(renderer.load(filtered.path));
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{text}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "plain string");
}

TEST(ChatTemplateTest, PreservesAbsentNullEmptyAndStringContent)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- for message in messages -%}
{%- if 'content' not in message -%}absent
{%- else if message.content is none -%}null
{%- else if message.content == '' -%}empty
{%- else -%}{{ message.content }}
{%- endif -%}{% if not loop.last %}|{% endif %}{%- endfor -%})JINJA"}};

    rt::Message absent{"assistant", {}};
    absent.hasContent = false;
    rt::Message nullContent{"assistant", {}};
    nullContent.contentIsNull = true;
    rt::Message empty{"assistant", {{"text", ""}}};
    rt::Message value{"assistant", {{"text", "answer"}}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{absent, nullContent, empty, value}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "absent|null|empty|answer");
}

TEST(ChatTemplateTest, RejectsNullContentWithContentBlocks)
{
    TemporaryTemplate model{{"chat_template.jinja", "{{ messages[0].content }}"}};
    rt::Message message{"assistant", {{"text", "conflict"}}};
    message.contentIsNull = true;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    EXPECT_FALSE(renderer.apply(rt::LLMGenerationRequest::Request{{message}}, formatted, {}));
}

TEST(ChatTemplateTest, PreservesToolArgumentRepresentation)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- for call in messages[0].tool_calls -%}
{%- if call.function.arguments is string -%}string={{ call.function.arguments }}
{%- else if call.function.arguments is mapping -%}object={{ call.function.arguments.city }}
{%- endif -%}{% if not loop.last %}|{% endif %}{%- endfor -%})JINJA"}};
    rt::Message assistant{"assistant", {}};
    assistant.contentIsNull = true;
    assistant.toolCalls = {{"call_1", "function", "weather", R"({"city":"Paris"})", true},
        {"call_2", "function", "weather", R"({"city":"Berlin"})", false}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{assistant}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, R"(string={"city":"Paris"}|object=Berlin)");
}

TEST(ChatTemplateTest, ExposesProviderDateFunctionAsDefined)
{
    TemporaryTemplate model{{"chat_template.jinja",
        "{% if strftime_now is defined %}{{ strftime_now('%Y') }}{% else %}missing{% endif %}"}};
    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "hello"}}}}}, formatted, {}));
    EXPECT_NE(formatted.formattedCompleteRequest, "missing");
    EXPECT_EQ(formatted.formattedCompleteRequest.size(), 4);
}

TEST(ChatTemplateTest, FlattensToolResultsForStructuredProviderTemplates)
{
    TemporaryTemplate model{{"chat_template.jinja",
        "{% for message in messages %}{% if message.role == 'tool' %}{{ message.content }}{% else %}"
        "{% for item in message.content %}{{ item.text }}{% endfor %}{% endif %}{% endfor %}"}};
    rt::Message user{"user", {{"text", "question"}}};
    rt::Message tool{"tool", {{"text", "line one"}, {"text", "line two"}}};
    tool.contentIsArray = true;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{user, tool}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "questionline one\nline two");
}

TEST(ChatTemplateTest, AppliesCommonProviderJinjaSyntaxDirectly)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- set ns = namespace(found=false, count=0) -%}{%- set ns.found = true -%}{%- if ns.found and missing is undefined and not optional and messages[0]['role'] == 'system' -%}{{ messages[0]['content'] }}|{%- endif -%}{%- for message in messages -%}{{ message.role }}{% if not loop.last %},{% endif %}{%- for tool_call in message.tool_calls -%}{%- if tool_call.function is defined -%}{%- set tool_call = tool_call.function -%}{%- endif -%}:{{ tool_call.name }}={{ tool_call.arguments | tojson }}{%- endfor -%}{%- endfor -%}{%- if add_generation_prompt -%}|assistant{%- endif -%})JINJA"}};

    rt::Message assistant{"assistant", {}};
    assistant.contentIsNull = true;
    assistant.toolCalls.push_back({"call_1", "function", "lookup", R"({"query":"weather"})", false});
    rt::LLMGenerationRequest::Request request;
    request.messages
        = {rt::Message{"system", {{"text", "system"}}}, rt::Message{"user", {{"text", "hello"}}}, std::move(assistant)};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(request, formatted, {}));
    EXPECT_EQ(
        formatted.formattedCompleteRequest, R"(system|system,user,assistant:lookup={"query": "weather"}|assistant)");
}

TEST(ChatTemplateTest, AppliesLiteralNewlinesFromProviderJinja)
{
    TemporaryTemplate model{{"chat_template.jinja", "{{ '<|im_start|>user\n' }}{{ messages[0]['content'] }}"}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "hello"}}}}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "<|im_start|>user\nhello");
}

TEST(ChatTemplateTest, AppliesProviderJinjaLoop)
{
    TemporaryTemplate model{{"chat_template.jinja", "{% for index in range(2) %}<video>\n{% endfor %}"}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(
        rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "describe"}}}}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "<video>\n<video>\n");
}

TEST(ChatTemplateTest, AppliesProviderMacroDirectly)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- set count = namespace(value=0) -%}{%- macro content(message, prefix='') -%}{%- set count.value = count.value + 1 -%}{{ prefix ~ message.content }}{%- endmacro -%}{{ 'value=' + content(messages[0], prefix='user:') }}|{{ count.value }})JINJA"}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "hello"}}}}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "value=user:hello|1");
}

TEST(ChatTemplateTest, PreservesUndefinedArgumentsPassedToProviderMacros)
{
    TemporaryTemplate model{{"chat_template.jinja",
        "{% macro state(value) %}{{ 'present' if value is defined else 'missing' }}{% endmacro %}"
        "{{ state(tools[0].function.parameters['required']) }}"}};
    rt::ToolDefinition tool;
    tool.name = "weather";
    tool.parameters = R"({"type":"object"})";

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    chat_template::ChatTemplate::Options options;
    options.tools = {tool};
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{{"user", {{"text", "Weather?"}}}}}, formatted, options));
    EXPECT_EQ(formatted.formattedCompleteRequest, "missing");
}

TEST(ChatTemplateTest, AppliesProviderCollectionsAndLoopControlsDirectly)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- set values = [{'name': 'skip'}, {'name': messages[0].role}, {'name': 'unused'}] -%}{%- for value in values -%}{%- if value.name == 'skip' -%}{%- continue -%}{%- endif -%}{%- generation -%}{{ {'z': value.name, 'a': 1} | tojson(separators=(',', ':')) }}{%- endgeneration -%}{%- break -%}{%- endfor -%}{%- if 2 in (1, 2) -%}|tuple{%- endif -%})JINJA"}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "hello"}}}}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, R"({"z":"user","a":1}|tuple)");
}

TEST(ChatTemplateTest, PreservesProviderDefaultForUnsetReasoningEffort)
{
    TemporaryTemplate model{{"chat_template.jinja",
        "{% if reasoning_effort is defined %}{{ reasoning_effort }}{% else %}provider-default{% endif %}"}};
    rt::LLMGenerationRequest::Request request{{rt::Message{"user", {{"text", "hello"}}}}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(request, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "provider-default");

    chat_template::ChatTemplate::Options options;
    options.reasoningEffort = "high";
    ASSERT_TRUE(renderer.apply(request, formatted, options));
    EXPECT_EQ(formatted.formattedCompleteRequest, "high");
}

TEST(ChatTemplateTest, AppliesQwen38ReasoningEffortContract)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- if enable_thinking is false -%}off|{{- '<think>\n\n</think>\n\n' -}}
{%- else -%}
{%- set effort = reasoning_effort|default('xhigh') -%}
{%- if effort not in ('xhigh', 'medium', 'low') -%}
{{- raise_exception('unsupported reasoning effort') -}}
{%- endif -%}
{{- effort -}}|{{- '<think>\n' -}}
{%- endif -%})JINJA"}};
    rt::LLMGenerationRequest::Request request{{rt::Message{"user", {{"text", "hello"}}}}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    chat_template::ChatTemplate::Options options;

    ASSERT_TRUE(renderer.apply(request, formatted, options));
    EXPECT_EQ(formatted.formattedCompleteRequest, "off|<think>\n\n</think>\n\n");

    options.enableThinking = true;
    ASSERT_TRUE(renderer.apply(request, formatted, options));
    EXPECT_EQ(formatted.formattedCompleteRequest, "xhigh|<think>\n");

    for (auto const* effort : {"xhigh", "medium", "low"})
    {
        options.reasoningEffort = effort;
        ASSERT_TRUE(renderer.apply(request, formatted, options));
        EXPECT_EQ(formatted.formattedCompleteRequest, std::string(effort) + "|<think>\n");
    }

    options.reasoningEffort = "unsupported";
    EXPECT_FALSE(renderer.apply(request, formatted, options));
}

TEST(ChatTemplateTest, LeavesEnableThinkingUndefinedForQwen3Omni)
{
    // Generation-prompt tail of the Qwen3-Omni provider template, verbatim.
    constexpr char kTail[] = R"JINJA({%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}
{%- if enable_thinking is defined and enable_thinking is false -%}{{- '<think>\n\n</think>\n\n' -}}{%- endif -%}
{%- endif -%})JINJA";
    rt::LLMGenerationRequest::Request request{{rt::Message{"user", {{"text", "hello"}}}}};
    rt::LLMGenerationRequest::FormattedRequest formatted;

    {
        TemporaryTemplate model{{"chat_template.jinja", kTail}, {"config.json", R"({"model":"qwen3_omni_moe_text"})"}};
        chat_template::ChatTemplate renderer;
        ASSERT_TRUE(renderer.load(model.path));
        ASSERT_TRUE(renderer.apply(request, formatted, {}));
        EXPECT_EQ(formatted.formattedCompleteRequest, "<|im_start|>assistant\n");
    }
    {
        TemporaryTemplate model{{"chat_template.jinja", kTail}, {"config.json", R"({"model":"qwen3_moe"})"}};
        chat_template::ChatTemplate renderer;
        ASSERT_TRUE(renderer.load(model.path));
        ASSERT_TRUE(renderer.apply(request, formatted, {}));
        EXPECT_EQ(formatted.formattedCompleteRequest, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
    {
        TemporaryTemplate model{{"chat_template.jinja", kTail}};
        chat_template::ChatTemplate renderer;
        ASSERT_TRUE(renderer.load(model.path));
        ASSERT_TRUE(renderer.apply(request, formatted, {}));
        EXPECT_EQ(formatted.formattedCompleteRequest, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
}

TEST(ChatTemplateTest, ExposesReasoningUnderCurrentAndLegacyProviderKeys)
{
    TemporaryTemplate model{{"chat_template.jinja", "{{ messages[0].reasoning }}|{{ messages[0].reasoning_content }}"}};
    rt::Message assistant{"assistant", {{"text", "answer"}}};
    assistant.reasoningContent = "analysis";
    assistant.hasReasoningContent = true;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{assistant}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "analysis|analysis");
}

TEST(ChatTemplateTest, NormalizesDeveloperRoleLikeVllmWhenProviderDoesNotSupportIt)
{
    TemporaryTemplate model{
        {"chat_template.jinja", "{% for message in messages %}{{ message.role }}={{ message.content }}|{% endfor %}"}};
    rt::Message system{"system", {{"text", "Base policy."}}};
    rt::Message user{"user", {{"text", "Hello."}}};
    rt::Message developer{"developer", {{"text", "Use tools."}}};
    rt::Message secondUser{"user", {{"text", "Weather?"}}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{system, user, developer, secondUser}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "system=Base policy.\n\nUse tools.|user=Hello.|user=Weather?|");
}

TEST(ChatTemplateTest, PreservesDeveloperRoleWhenProviderSupportsIt)
{
    TemporaryTemplate model{{"chat_template.jinja",
        "{% for message in messages %}{% if message.role == 'developer' %}developer={{ message.content }}{% endif %}"
        "{% endfor %}"}};
    rt::Message developer{"developer", {{"text", "Use tools."}}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{developer}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "developer=Use tools.");
}

TEST(ChatTemplateTest, AppliesProviderSequenceAttributeFilters)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- set values = [{'type': 'text', 'value': 'a'}, {'type': 'image', 'value': 'x'}, {'type': 'text', 'value': 'a'}, {'value': 'missing'}] -%}
{{- values | selectattr('type', 'equalto', 'text') | map(attribute='value') | unique | list | join(',') -}}|
{{- values | rejectattr('type', 'equalto', 'image') | selectattr('type', 'defined') | list | length -}})JINJA"}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "hello"}}}}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "a|2");
}

TEST(ChatTemplateTest, PreservesAbsentToolCallsAndFiltersNullTools)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- if 'tool_calls' in messages[0] -%}calls{%- else -%}no-calls{%- endif -%}|
{{- tools | selectattr('type', 'equalto', 'function') | list | length -}})JINJA"}};
    rt::Message message{"user", {{"text", "hello"}}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{message}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "no-calls|0");

    message.hasToolCalls = true;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{message}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "calls|0");
}

TEST(ChatTemplateTest, UsesPythonRepresentationForProviderCollections)
{
    TemporaryTemplate model{
        {"chat_template.jinja", "{% for item in messages[0].content %}{% endfor %}{{ messages[0].content }}"}};
    rt::Message message{"user", {{"text", "it's ready"}}};
    message.contentIsArray = true;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{message}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, R"([{'type': 'text', 'text': "it's ready"}])");
}

TEST(ChatTemplateTest, PreservesProviderUnknownStringEscapes)
{
    TemporaryTemplate model{{"chat_template.jinja", R"JINJA({{ '<\s>' }})JINJA"}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "hello"}}}}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, R"(<\s>)");
}

TEST(ChatTemplateTest, RejectsUnknownProviderFunctionDuringLoad)
{
    TemporaryTemplate model{{"chat_template.jinja", "{{ unsupported_provider_function(messages) }}"}};

    chat_template::ChatTemplate renderer;
    EXPECT_FALSE(renderer.load(model.path));
}

TEST(ChatTemplateTest, AppliesPhi4MMRawProcessorContract)
{
    TemporaryTemplate model{
        {"chat_template.jinja", "{{ messages.0.content }}"}, {"chat_template.processor", "phi4mm\n"}};

    rt::Message message{"user", {{"image", ""}, {"text", "Describe it."}, {"audio", ""}}};
    message.contentIsArray = true;
    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{std::move(message)}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "<|endoftext10|>Describe it.<|endoftext11|>");
}

TEST(ChatTemplateTest, RejectsMediaOutsidePhi4MMProcessorContract)
{
    TemporaryTemplate model{
        {"chat_template.jinja", "{{ messages.0.content }}"}, {"chat_template.processor", "phi4mm\n"}};

    rt::Message message{"user", {{"video", ""}}};
    message.contentIsArray = true;
    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    EXPECT_FALSE(renderer.apply(rt::LLMGenerationRequest::Request{{std::move(message)}}, formatted, {}));
}

TEST(ChatTemplateTest, AppliesLegacyInternVLProcessorContract)
{
    TemporaryTemplate model{
        {"chat_template.jinja", "{{ messages.0.content }}"}, {"chat_template.processor", "internvl\n"}};
    rt::Message message{"user", {{"image", ""}, {"text", "Describe it."}, {"video", ""}}};
    message.contentIsArray = true;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{message}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "<IMG_CONTEXT>\nDescribe it.<video>\n");
}

TEST(ChatTemplateTest, RejectsMediaWithoutProviderOrNativeProcessorContract)
{
    TemporaryTemplate model{{"chat_template.jinja", "{{ messages.0.content }}"}};
    rt::Message message{"user", {{"image", ""}}};
    message.contentIsArray = true;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    EXPECT_FALSE(renderer.apply(rt::LLMGenerationRequest::Request{{message}}, formatted, {}));
}

TEST(ChatTemplateTest, AppliesNemotronOmniProcessorContract)
{
    TemporaryTemplate model{
        {"chat_template.jinja", "{{ messages.0.content }}"}, {"chat_template.processor", "nemotron_omni\n"}};
    rt::Message message{"user", {{"image", ""}, {"text", "What is in this image?"}}};
    message.contentIsArray = true;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(rt::LLMGenerationRequest::Request{{message}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "<image>What is in this image?");
}

TEST(ChatTemplateTest, UsesNativeRendererOnlyWhenCheckpointHasNoJinja)
{
    TemporaryTemplate model{{"chat_template.model", "qwen3_tts\n"}};
    rt::LLMGenerationRequest::Request request;
    request.messages = {rt::Message{"user", {{"text", "Speak this"}}}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(request, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, "<|im_start|>user\nSpeak this<|im_end|>\n<|im_start|>assistant\n");
}

TEST(ChatTemplateTest, AppliesQwen3AsrNativeContract)
{
    TemporaryTemplate model{{"chat_template.model", "qwen3_asr\n"}};
    rt::Message audio{"user", {{"audio", ""}}};
    audio.contentIsArray = true;
    rt::LLMGenerationRequest::Request request;
    request.messages = {rt::Message{"system", {{"text", "Transcribe English."}}}, std::move(audio)};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(request, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest,
        "<|im_start|>system\nTranscribe English.<|im_end|>\n<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n");
    EXPECT_EQ(formatted.formattedSystemPrompt, "<|im_start|>system\nTranscribe English.<|im_end|>\n<|im_start|>user\n");
}

TEST(ChatTemplateTest, AppliesMultiturnReasoningToolAndMediaContract)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({%- if tools -%}tools={{ tools | tojson }}|{%- endif -%}
{%- for message in messages -%}
{{- message.role }}:
{%- if message.reasoning is defined -%}[reasoning={{ message.reasoning }}]{%- endif -%}
{%- if message.role == 'tool' -%}[result={{ message.tool_call_id }}:{{ message.content }}]
{%- else if message.content is none -%}[null]
{%- else if message.content is string -%}{{ message.content }}
{%- else -%}
{%- for item in message.content -%}
{%- if item.type == 'text' -%}{{ item.text }}{%- else -%}<{{ item.type }}>{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- for call in message.tool_calls -%}
[call={{ call.id }}:{{ call.function.name }}:{{ call.function.arguments | tojson }}]
{%- endfor -%}|
{%- endfor -%}
assistant:{%- if enable_thinking -%}<think effort={{ reasoning_effort | default('xhigh') }}>{%- else -%}<think></think>{%- endif -%})JINJA"}};

    rt::Message user{"user", {{"image", ""}, {"text", "Describe"}, {"video", ""}, {"audio", ""}}};
    user.contentIsArray = true;
    rt::Message answer{"assistant", {{"text", "Done"}}};
    answer.reasoningContent = "Analysis";
    answer.hasReasoningContent = true;
    rt::Message call{"assistant", {}};
    call.contentIsNull = true;
    call.hasToolCalls = true;
    call.toolCalls.push_back({"call_1", "function", "weather", R"({"city":"Paris"})", false});
    rt::Message result{"tool", {{"text", "Sunny"}}};
    result.toolCallId = "call_1";

    rt::LLMGenerationRequest::Request request;
    request.messages = {std::move(user), std::move(answer), std::move(call), std::move(result)};
    chat_template::ChatTemplate::Options options;
    options.enableThinking = true;
    options.reasoningEffort = "low";
    options.tools.push_back({"weather", "Get weather", R"({"type":"object"})", false});

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(request, formatted, options));
    EXPECT_EQ(formatted.formattedCompleteRequest,
        "tools=[{\"type\": \"function\", \"function\": {\"name\": \"weather\", \"description\": \"Get weather\", "
        "\"parameters\": {\"type\": \"object\"}}}]|user:<image>Describe<video><audio>|"
        "assistant:[reasoning=Analysis]Done|assistant:[null][call=call_1:weather:{\"city\": \"Paris\"}]|"
        "tool:[result=call_1:Sunny]|assistant:<think effort=low>");
}

TEST(ChatTemplateTest, ForwardsProviderToolSelectionOptions)
{
    TemporaryTemplate model{{"chat_template.jinja",
        "{{ tool_choice | tojson }}|{{ parallel_tool_calls | string }}|{{ add_generation_prompt | string }}"}};
    chat_template::ChatTemplate::Options options;
    options.toolChoice.mode = rt::ToolChoice::Mode::kFunction;
    options.toolChoice.functionName = "weather";
    options.parallelToolCalls = false;
    options.addGenerationPrompt = false;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(
        rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "Weather?"}}}}}, formatted, options));
    EXPECT_EQ(
        formatted.formattedCompleteRequest, R"({"type": "function", "function": {"name": "weather"}}|False|False)");
}

TEST(ChatTemplateTest, AppliesProviderFiltersUsedBySupportedFamilies)
{
    TemporaryTemplate model{{"chat_template.jinja",
        R"JINJA({{ '' | default('fallback', true) }}|{{ '  text  ' | trim | upper }}|{{ {'z': 2, 'a': 1} | dictsort | tojson }}|{{ ['a', 'b'] | map('upper') | join(',') }})JINJA"}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "hello"}}}}}, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest, R"(fallback|TEXT|[["a", 1], ["z", 2]]|A,B)");
}

TEST(ChatTemplateTest, AppliesAlpamayoNativeContract)
{
    TemporaryTemplate model{{"chat_template.model", "alpamayo\n"}};
    rt::Message user{"user", {{"image", ""}, {"text", "Drive"}, {"video", ""}, {"trajectory", ""}}};
    user.contentIsArray = true;
    rt::LLMGenerationRequest::Request request;
    request.messages = {std::move(user)};
    request.pastTrajectory = std::vector<rt::PastTrajectoryPoint>{{0.F, 0.F, 0.F}, {1.F, 2.F, 3.F}};

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(renderer.apply(request, formatted, {}));
    EXPECT_EQ(formatted.formattedCompleteRequest,
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Drive"
        "<|vision_start|><|video_pad|><|vision_end|><|traj_history_start|>"
        "<|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|><|traj_history|>"
        "<|traj_history_end|><|im_end|>\n<|im_start|>assistant\n");
}

TEST(ChatTemplateTest, RejectsTrajectoryWithoutRuntimeData)
{
    TemporaryTemplate model{{"chat_template.model", "alpamayo\n"}};
    rt::Message user{"user", {{"trajectory", ""}}};
    user.contentIsArray = true;

    chat_template::ChatTemplate renderer;
    ASSERT_TRUE(renderer.load(model.path));
    rt::LLMGenerationRequest::FormattedRequest formatted;
    EXPECT_FALSE(renderer.apply(rt::LLMGenerationRequest::Request{{std::move(user)}}, formatted, {}));
}

TEST(ChatTemplateTest, AppliesPreformattedPromptWithoutLoadedTemplate)
{
    TemporaryTemplate model{};
    chat_template::ChatTemplate renderer;
    EXPECT_FALSE(renderer.load(model.path));

    chat_template::ChatTemplate::Options options;
    options.applyTemplate = false;
    rt::LLMGenerationRequest::FormattedRequest formatted;
    ASSERT_TRUE(
        renderer.apply(rt::LLMGenerationRequest::Request{{rt::Message{"user", {{"text", "preformatted prompt"}}}}},
            formatted, options));
    EXPECT_EQ(formatted.formattedCompleteRequest, "preformatted prompt");
}
