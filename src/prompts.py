"""Build the exact model input for each mode (think, default, reinit).

Purpose : produce, by hand, the full string the model will see, so it can be logged,
          diffed and checked. We never let a server apply the chat template, because
          Qwen3.5's template silently turns thinking OFF unless enable_thinking=True.
Inputs  : a question (text + correct answer + 3 wrong answers, from data/gpqa_diamond.jsonl),
          a sample index, and the [prompt] section of configs/control.toml.
Outputs : - shuffle_options -> options in display order, gold letter, permutation
          - build_text      -> the complete prompt string (system turn if any + user turn +
                               assistant prefix)
          - tokens_through  -> how many output tokens it took to finish a text marker
                               (counts the reinit <analysis> block)
Pipeline: data/gpqa_diamond.jsonl -> [prompts] -> (later) src/generate.py -> SGLang.
          src/data.py uses it to print example prompts (milestone 1);
          check_against_tokenizer is the template-parity check (milestone 3).

Template source: Qwen/Qwen3.5-2B chat_template.jinja @ 15852e8c (read 2026-11-13):
    system turn     : '<|im_start|>system\\n' + content|trim + '<|im_end|>\\n'  (only if a
                      system message is given; it must come first)
    user turn       : '<|im_start|>user\\n' + content|trim + '<|im_end|>\\n'
    generation start: '<|im_start|>assistant\\n' then
                      '<think>\\n'               if enable_thinking is true
                      '<think>\\n\\n</think>\\n\\n'  otherwise
    No system turn is inserted when no system message and no tools are given.

Modes:
  think   - the control user prompt; thinking ON  (template enable_thinking=True)
  default - the control user prompt; thinking OFF (template enable_thinking=False)
  reinit  - "reinitialized thinking": thinking OFF, plus a system persona, a user prompt that
            asks for a step-by-step analysis inside <analysis></analysis>, and the assistant
            turn pre-started with "<analysis>\\n" (the official thinking-off prefix + our tag).
            The model reasons in a fresh, plain-text block instead of Qwen's trained <think>.
            Prompt text: the user's draft (2026-11-15), with fixes agreed that day.
"""
from __future__ import annotations

import hashlib
import random
import string

MODES = ("think", "default", "reinit")

# The two generation prefixes the official template can produce.
THINK_ON_PREFIX = "<think>\n"                # enable_thinking=True
THINK_OFF_PREFIX = "<think>\n\n</think>\n\n"  # enable_thinking=False (the template default)

# The reinit analysis block. Plain text, NOT special tokens: "<analysis>" is split into
# several ordinary tokens by the tokenizer, unlike <think> (one special token, id 248068).
ANALYSIS_OPEN = "<analysis>"
ANALYSIS_CLOSE = "</analysis>"

# Start of the assistant turn per mode. think/default are copied character-for-character
# from the template; reinit appends the opened analysis block to the thinking-off prefix, so
# the block always starts (as think mode always starts <think>) and the model writes its
# analysis next.
ASSISTANT_PREFIX = {
    "think": THINK_ON_PREFIX,                          # open think block: reasoning next
    "default": THINK_OFF_PREFIX,                       # closed empty block: answers directly
    "reinit": THINK_OFF_PREFIX + ANALYSIS_OPEN + "\n",  # thinking off, analysis block opened
}

# The answer-format instruction recommended by Qwen for multiple-choice benchmarking,
# copied verbatim (backticks included) from the Qwen3.5-2B model card README @ 15852e8c,
# line 983, "Standardize Output Format": "Multiple-Choice Questions: Add the following
# JSON structure to the prompt to standardize responses: ...". Using the model's own
# recommended format makes "the model failed the format" less likely than a custom one.
MCQ_FORMAT_INSTRUCTION = (
    'Please show your choice in the `answer` field with only the choice letter, '
    'e.g., `"answer": "C"`.'
)

# Short, domain-spanning examples teach the answer format without using GPQA questions.
# These examples are shared by Think and Default so their user prompts remain identical.
FEW_SHOT_EXAMPLES = (
    "Example 1\n"
    "Question: A 2.0 kg object experiences a net force of 6.0 N. What is its acceleration?\n"
    "A) 0.33 m/s^2\nB) 3.0 m/s^2\nC) 8.0 m/s^2\nD) 12.0 m/s^2\n"
    "Example response:\n"
    '{{"answer": "B"}}\n\n'
    "Example 2\n"
    "Question: An atom has 17 protons and 18 neutrons. What is its atomic number?\n"
    "A) 17\nB) 18\nC) 35\nD) 1\n"
    "Example response:\n"
    '{{"answer": "A"}}\n\n'
    "Example 3\n"
    "Question: Which molecule carries amino acids to the ribosome during translation?\n"
    "A) mRNA\nB) rRNA\nC) DNA\nD) tRNA\n"
    "Example response:\n"
    '{{"answer": "D"}}\n\n'
)

# The user-turn text of Think and Default. These modes share the exact same user prompt;
# their only experimental difference is the assistant prefix (thinking enabled vs disabled).
QUESTION_PROMPT = (
    "Answer the following multiple-choice question.\n\n"
    "Use these examples as a guide to the required final-answer format.\n\n"
    + FEW_SHOT_EXAMPLES
    + "Now answer the following question.\n\n"
    "Question: {question}\n\n{options}\n\n"
    + MCQ_FORMAT_INSTRUCTION
)

# Hash of the prompt template + prefixes. Logged with every generation so two runs can
# be checked for identical prompting without diffing source code.
# --- reinit mode ------------------------------------------------------------------
# System turn: the persona only. Qwen3.5 has no default system prompt, so think/default
# send none; reinit adds this one (the user's choice, 2026-11-15).
REINIT_SYSTEM_PROMPT = (
    "You are a PhD-level expert in chemistry, biology, and physics, skilled in solving "
    "multiple-choice problems."
)

# Reinit examples demonstrate the requested three-step analysis, the closing marker, and
# the same single-line JSON answer format used in Think and Default. They are synthetic
# examples, not questions drawn from GPQA Diamond.
REINIT_FEW_SHOT_EXAMPLES = (
    "Example 1 (physics)\n"
    "Question: A 2.0 kg object experiences a net force of 6.0 N. What is its acceleration?\n"
    "A) 0.33 m/s^2\nB) 3.0 m/s^2\nC) 8.0 m/s^2\nD) 12.0 m/s^2\n"
    "<analysis>\n"
    "1. Deconstruct: The problem gives the net force and mass and asks for acceleration.\n"
    "2. Derivation: Newton's second law gives a = F/m = 6.0/2.0 = 3.0 m/s^2, which is option B.\n"
    "3. Verification: N/kg is equivalent to m/s^2, so the units and calculated value match option B.\n"
    "</analysis>\n"
    '{{"answer": "B"}}\n\n'
    "Example 2 (chemistry)\n"
    "Question: An atom has 17 protons and 18 neutrons. What is its atomic number?\n"
    "A) 17\nB) 18\nC) 35\nD) 1\n"
    "<analysis>\n"
    "1. Deconstruct: Atomic number is determined by the number of protons; neutrons do not set it.\n"
    "2. Derivation: The atom has 17 protons, so its atomic number is 17, which is option A.\n"
    "3. Verification: The mass number would be 17 + 18 = 35, but the question asks for atomic number. Option A is correct.\n"
    "</analysis>\n"
    '{{"answer": "A"}}\n\n'
    "Example 3 (biology)\n"
    "Question: Which molecule carries amino acids to the ribosome during translation?\n"
    "A) mRNA\nB) rRNA\nC) DNA\nD) tRNA\n"
    "<analysis>\n"
    "1. Deconstruct: The question asks which molecule delivers amino acids to the ribosome during protein synthesis.\n"
    "2. Derivation: Transfer RNA (tRNA) carries specific amino acids to the ribosome, so option D is correct.\n"
    "3. Verification: mRNA carries the codon sequence, while tRNA brings amino acids; therefore D matches the described role.\n"
    "</analysis>\n"
    '{{"answer": "D"}}\n\n'
)

# Reinit's assistant prefix already opens <analysis>. The user prompt tells the model to
# continue that block, close it once, and place the final answer outside it.
REINIT_QUESTION_PROMPT = (
    "Your task is to solve a highly challenging, graduate-level multiple-choice problem in "
    "chemistry, biology, or physics.\n\n"
    "Continue writing inside the <analysis> block that has already been opened for you. "
    "Do not write another opening <analysis> tag.\n"
    "Complete these three steps in order inside that block:\n"
    "1. Deconstruct: identify the scientific principles, constraints, specific conditions, "
    "and core theoretical mechanisms involved.\n"
    "2. Derivation: work through the problem systematically, showing relevant calculations, "
    "mechanism deductions, or theoretical reasoning.\n"
    "3. Verification: check the result, including logic, units, and assumptions.\n\n"
    "After step 3, generate the closing tag </analysis> exactly once, on its own line. "
    "Do not leave the analysis block open or start another analysis block. "
    "Put the final answer only after </analysis>.\n\n"
    "The examples show the required analysis structure and final-answer format.\n\n"
    + REINIT_FEW_SHOT_EXAMPLES
    + "Now solve the following question.\n\n"
    "Question: {question}\n\n{options}\n\n"
    + MCQ_FORMAT_INSTRUCTION
)

# Per-mode (system prompt or None, user template).
MODE_PROMPTS = {
    "think": (None, QUESTION_PROMPT),
    "default": (None, QUESTION_PROMPT),
    "reinit": (REINIT_SYSTEM_PROMPT, REINIT_QUESTION_PROMPT),
}

PROMPT_HASH = hashlib.sha256("\x00".join(
    f"{m}\x00{MODE_PROMPTS[m][0] or ''}\x00{MODE_PROMPTS[m][1]}\x00{ASSISTANT_PREFIX[m]}" for m in MODES
).encode()).hexdigest()[:16]


def shuffle_seed(salt: str, qid: str, sample_idx: int) -> str:
    """The seed string for one (question, sample). Both modes use the same seed, so
    all modes see the options in the same order and can be compared pairwise."""
    return f"{salt}|{qid}|{sample_idx}"


def shuffle_options(qid: str, sample_idx: int, correct: str, wrong: list[str],
                    salt: str) -> tuple[list[str], str, list[int]]:
    """Shuffle the 4 options deterministically.

    Returns (options, gold, order):
      options : option texts in display order (A, B, C, D)
      gold    : letter of the correct option, e.g. "C"
      order   : order[i] = index into [correct] + wrong of the option shown at position i.
                Index 0 is always the correct answer. Logged instead of option text, so
                the shuffle is recorded without copying GPQA text into logs.
    """
    pool = [correct] + list(wrong)
    order = list(range(len(pool)))
    # random.Random(str) hashes the string with SHA-512 (seed version 2): the result is
    # the same on every machine and Python version >= 3.2, unlike hash(str).
    random.Random(shuffle_seed(salt, qid, sample_idx)).shuffle(order)
    options = [pool[i] for i in order]
    gold = string.ascii_uppercase[order.index(0)]  # where the correct answer landed
    return options, gold, order


def user_text(question: str, options: list[str], mode: str = "think") -> str:
    """Fill the mode's user template: lettered options 'A) ...' one per line. (.format parses
    only the template, so braces inside question text, e.g. LaTeX, are inserted unchanged.)"""
    _check_mode(mode)
    letters = string.ascii_uppercase[: len(options)]
    opt_block = "\n".join(f"{L}) {o}" for L, o in zip(letters, options))
    return MODE_PROMPTS[mode][1].format(question=question, options=opt_block)


def system_text(mode: str) -> str | None:
    """The mode's system message, or None (no system turn)."""
    _check_mode(mode)
    return MODE_PROMPTS[mode][0]


def enable_thinking(mode: str) -> bool:
    """The template flag that each mode corresponds to (used only for the parity check)."""
    _check_mode(mode)
    return mode == "think"


def official_suffix(mode: str) -> str:
    """Text we add AFTER the official template's generation prefix ("" except reinit)."""
    base = THINK_ON_PREFIX if enable_thinking(mode) else THINK_OFF_PREFIX
    assert ASSISTANT_PREFIX[mode].startswith(base)
    return ASSISTANT_PREFIX[mode][len(base):]


def build_messages(question: str, options: list[str], mode: str = "think") -> list[dict]:
    """The chat-message form of the same input (for tokenizer.apply_chat_template)."""
    msgs = [{"role": "system", "content": system_text(mode)}] if system_text(mode) else []
    return msgs + [{"role": "user", "content": user_text(question, options, mode)}]


def build_text(mode: str, question: str, options: list[str]) -> str:
    """The complete raw input string for a mode: [system turn] + user turn + start of
    assistant turn."""
    _check_mode(mode)
    system = system_text(mode)
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return (
        head + "<|im_start|>user\n" + user_text(question, options, mode) + "<|im_end|>\n"
        "<|im_start|>assistant\n" + ASSISTANT_PREFIX[mode]
    )


def tokens_through(decode, ids: list[int], marker: str) -> int | None:
    """Number of leading tokens needed for `marker` to appear in decode(ids[:k]); None if
    it never appears. Used for reinit's n_analysis_tokens (marker = ANALYSIS_CLOSE), where
    the marker is several ordinary tokens, so it can't be found by a single token id.

    `decode` maps a list of ids to text (the pinned tokenizer's decode, special tokens kept).
    "marker in decode(ids[:k])" can only switch from False to True as k grows, so a binary
    search needs ~17 decodes even for an 81,920-token output (a linear scan would need up
    to 81,920)."""
    if marker not in decode(ids):
        return None
    lo, hi = 1, len(ids)  # invariant: the answer is in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        if marker in decode(ids[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo


def check_against_tokenizer(tokenizer, mode: str, question: str, options: list[str]) -> dict:
    """Milestone 3 (template parity): compare build_text with the official template.

    `tokenizer` is a transformers tokenizer loaded from Qwen/Qwen3.5-2B at the pinned
    revision. Returns a report; report["match"] must be True for every mode.
    """
    # reinit: the official template has no analysis block, so the expected text is the
    # official thinking-off output + our "<analysis>\\n" (official_suffix).
    official = tokenizer.apply_chat_template(
        build_messages(question, options, mode), tokenize=False,
        add_generation_prompt=True, enable_thinking=enable_thinking(mode),
    ) + official_suffix(mode)
    ours = build_text(mode, question, options)
    ids = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    return {"mode": mode, "match": official == ours,
            "ids_match": ids(official) == ids(ours),
            "official_tail": official[-40:], "ours_tail": ours[-40:]}


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
