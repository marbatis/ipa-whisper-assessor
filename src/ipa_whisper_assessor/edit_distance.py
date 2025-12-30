from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Op = Literal["match", "sub", "ins", "del"]


@dataclass(frozen=True)
class EditOp:
    op: Op
    expected: Optional[str] = None
    predicted: Optional[str] = None


def levenshtein_ops(expected: list[str], predicted: list[str]) -> list[EditOp]:
    """
    Levenshtein DP that returns a stable edit script (prefers match/sub over indels on ties).
    """
    n = len(expected)
    m = len(predicted)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[Op]] = [["match"] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = 0 if expected[i - 1] == predicted[j - 1] else 1
            sub = dp[i - 1][j - 1] + cost_sub
            dele = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = min(sub, dele, ins)
            dp[i][j] = best
            if best == sub:
                back[i][j] = "match" if cost_sub == 0 else "sub"
            elif best == dele:
                back[i][j] = "del"
            else:
                back[i][j] = "ins"

    ops: list[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = back[i][j]
        if step in {"match", "sub"}:
            ops.append(
                EditOp(step, expected=expected[i - 1], predicted=predicted[j - 1])  # type: ignore[arg-type]
            )
            i -= 1
            j -= 1
        elif step == "del":
            ops.append(EditOp("del", expected=expected[i - 1], predicted=None))
            i -= 1
        else:
            ops.append(EditOp("ins", expected=None, predicted=predicted[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def edit_counts(ops: list[EditOp]) -> tuple[int, int, int]:
    subs = sum(1 for o in ops if o.op == "sub")
    ins = sum(1 for o in ops if o.op == "ins")
    dels = sum(1 for o in ops if o.op == "del")
    return subs, ins, dels

