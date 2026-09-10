# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from examples.accuracy.scripts.calculate_correctness import (
    clean_text, parse_multi_choice_response)


def _parse(text):
    return parse_multi_choice_response(clean_text(text))


def test_parse_thinking_tail_single_letter_after_special_token():
    text = "<|channel>thought\nReasoning over A/B/C/D...\n\nD<eos>"

    assert _parse(text) == "D"


def test_parse_thinking_tail_explicit_correct_option():
    text = ("<|channel>thought\nLong reasoning that mentions A, B, C, and D.\n"
            "The correct option is C.<channel|>C<turn|>")

    assert _parse(text) == "C"


def test_parse_thinking_tail_result_marker():
    text = ('<|channel>thought\nLong reasoning.\n'
            'Constraint: "Reply with only the letter."\n\nResult: C<eos>')

    assert _parse(text) == "C"


def test_parse_short_thinking_tail_explicit_correct_option():
    assert _parse("Correct option is C.") == "C"


def test_parse_does_not_treat_so_rejected_option_as_answer():
    text = "Reasoning " * 20 + "So B is wrong, leaving C."

    assert _parse(text) == clean_text(text)


def test_parse_explicit_final_answer_beats_rejected_leading_option():
    text = "A is incorrect because of the constraint. The final answer is C."

    assert _parse(text) == "C"


def test_parse_explicit_answer_outside_tail_window():
    text = "The answer is B." + (" filler" * 900)

    assert _parse(text) == "B"


def test_parse_short_answer_unchanged():
    assert _parse("B<turn|>") == "B"


def test_parse_mmlu_pro_answer_letters():
    assert _parse("Final answer: J<eos>") == "J"
    assert _parse("(I)") == "I"


def test_parse_mmlu_pro_rejected_option_is_not_answer():
    text = "Option I is tempting, but it contradicts the premise; option J is also wrong."

    assert _parse(text) == clean_text(text)


def test_parse_answer_marker_with_both_is_and_colon():
    assert _parse("The correct answer is: B") == "B"
    assert _parse("The answer is: D<eos>") == "D"
    assert _parse("Result is: C") == "C"
    assert _parse("Option is: J") == "J"


def test_parse_answer_marker_with_interposed_question_reference():
    assert _parse(
        "The answer to the final question is:\n\n**C. Bitcoin**") == "C"
    assert _parse("The correct answer for the final question is: **A**") == "A"
    assert _parse("The answer to the last question is B") == "B"


def test_parse_qualifier_does_not_swallow_unrelated_clauses():
    # Guards the qualifier against being widened to a wildcard span, which
    # would match "... or there is a ..." and score that "a" as the answer.
    text = "Answer: None of the above (or there is a stronger counterexample)"

    assert _parse(text) == clean_text(text)
