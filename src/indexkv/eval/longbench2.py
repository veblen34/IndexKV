"""LongBench v2 official data/prompt/scoring adapter.

This module intentionally targets only THUDM/LongBench-v2. LongBench v1
QA/summarization metrics are outside the repository scope. Validation uses this
module and the synced benchmark resources under ``third_party/LongBench2``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from datasets import load_dataset


OFFICIAL_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "LongBench2"
PROMPT_0SHOT = OFFICIAL_ROOT / "prompts" / "0shot.txt"
PROMPT_COT = OFFICIAL_ROOT / "prompts" / "0shot_cot.txt"
PROMPT_COT_ANS = OFFICIAL_ROOT / "prompts" / "0shot_cot_ans.txt"
MODEL2PATH = OFFICIAL_ROOT / "config" / "model2path.json"
MODEL2MAXLEN = OFFICIAL_ROOT / "config" / "model2maxlen.json"

LEADERBOARD_TARGETS = {
    # https://longbench2.github.io/#leaderboard, last update shown there:
    # 2025-05-06. Columns are Overall/Easy/Hard/Short/Medium/Long.
    "Llama-3.1-8B-Instruct": {
        "direct": {
            "overall": 30.0,
            "easy": 30.7,
            "hard": 29.6,
            "short": 35.0,
            "medium": 27.9,
            "long": 25.9,
        },
        "cot": {
            "overall": 30.4,
            "easy": 36.5,
            "hard": 26.7,
            "short": 34.4,
            "medium": 31.6,
            "long": 21.3,
        },
    }
}


def load_model_maps() -> tuple[dict[str, str], dict[str, int]]:
    return json.loads(MODEL2PATH.read_text()), json.loads(MODEL2MAXLEN.read_text())


def load_prompt(cot: bool = False) -> str:
    return (PROMPT_COT if cot else PROMPT_0SHOT).read_text()


def load_cot_answer_prompt() -> str:
    return PROMPT_COT_ANS.read_text()


def load_longbench2(
    split: str = "train", *, revision: str | None = None
):
    """Load the official LongBench v2 dataset from HuggingFace."""
    return load_dataset(
        "THUDM/LongBench-v2", split=split, revision=revision
    )


def official_item(item: dict) -> dict:
    """Keep the exact fields used by THUDM/LongBench ``pred.py``."""
    fields = [
        "_id", "domain", "sub_domain", "difficulty", "length", "question",
        "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context",
    ]
    return {k: item[k] for k in fields}


def iter_official_items(
    split: str = "train", *, revision: str | None = None
) -> list[dict]:
    return [
        official_item(item)
        for item in load_longbench2(split=split, revision=revision)
    ]


def build_prompt(item: dict, *, cot: bool = False, context: str | None = None) -> str:
    """Render the official zero-shot or CoT prompt."""
    template = load_prompt(cot=cot)
    doc = item["context"] if context is None else context
    return (
        template.replace("$DOC$", doc.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def build_cot_answer_prompt(item: dict, cot_response: str, *,
                            context: str | None = None) -> str:
    """Render the official second-pass answer extraction prompt for CoT."""
    doc = item["context"] if context is None else context
    return (
        load_cot_answer_prompt().replace("$DOC$", doc.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
        .replace("$COT$", cot_response)
    )


def truncate_prompt(prompt: str, model_name: str, tokenizer) -> str:
    """Official middle truncation from THUDM/LongBench ``pred.py``."""
    _, maxlen_map = load_model_maps()
    max_len = maxlen_map[model_name]
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) <= max_len:
        return prompt
    input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def extract_answer(response: str) -> str | None:
    """Official LongBench v2 answer extractor."""
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None


@dataclass(frozen=True)
class LongBench2Score:
    overall: float
    easy: float
    hard: float
    short: float
    medium: float
    long: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "overall": self.overall,
            "easy": self.easy,
            "hard": self.hard,
            "short": self.short,
            "medium": self.medium,
            "long": self.long,
            "n": self.n,
        }


def _pct(num: float, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def score_predictions(preds: Iterable[dict], *, compensated: bool = False) -> LongBench2Score:
    """Aggregate predictions exactly like official ``result.py``.

    ``pred['judge']`` may be a bool or int. Scores are reported in percent.
    """
    rows = list(preds)
    easy = hard = short = medium = long = 0
    easy_acc = hard_acc = short_acc = medium_acc = long_acc = 0.0
    for pred in rows:
        acc = int(pred["judge"])
        if compensated and pred.get("pred") is None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred["length"] == "short":
            short += 1
            short_acc += acc
        elif pred["length"] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    total_acc = easy_acc + hard_acc
    return LongBench2Score(
        overall=_pct(total_acc, len(rows)),
        easy=_pct(easy_acc, easy),
        hard=_pct(hard_acc, hard),
        short=_pct(short_acc, short),
        medium=_pct(medium_acc, medium),
        long=_pct(long_acc, long),
        n=len(rows),
    )


def compare_to_leaderboard(
    score: LongBench2Score,
    model: str = "Llama-3.1-8B-Instruct",
    setting: Literal["direct", "cot"] = "direct",
    tolerance: float = 0.1,
) -> dict[str, dict[str, float | bool]]:
    """Compare local scores with the pinned LongBench v2 leaderboard row."""
    target = LEADERBOARD_TARGETS[model][setting]
    local = score.as_dict()
    out = {}
    for key, expected in target.items():
        actual = float(local[key])
        delta = round(actual - expected, 3)
        out[key] = {
            "local": actual,
            "leaderboard": expected,
            "delta": delta,
            "match": abs(delta) <= tolerance,
        }
    return out
