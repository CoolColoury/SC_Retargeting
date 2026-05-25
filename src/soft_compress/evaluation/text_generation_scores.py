"""
Sentence-level BLEU / ROUGE for reconstructed text.

NLTK sentence_bleu assigns zero mass when higher-order n-grams are missing; we
always apply additive smoothing (NLTK Chen and Cherry method1) so short spans
and near-misses stay numeric.

When the tokenized hypothesis and reference differ in length (common under open
greedy decoding), we evaluate BLEU on the contiguous same-length substring of the
longer sequence that maximizes smoothed 4-component BLEU. The shorter sequence is
held fixed. Search uses a stride when the longer side is long, then a local
refinement pass around the argmax.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

_SMOOTH = SmoothingFunction().method1
_WEIGHTS4 = (0.25, 0.25, 0.25, 0.25)


def safe_tokenize(text: str) -> List[str]:
    """Offline-safe tokenization (no punkt download)."""
    try:
        return wordpunct_tokenize(text)
    except Exception:
        return text.split()


def _sentence_bleu(ref: Sequence[str], hyp: Sequence[str], weights: Tuple[float, ...]) -> float:
    if len(ref) == 0 or len(hyp) == 0:
        return 0.0
    return float(
        sentence_bleu([list(ref)], list(hyp), weights=weights, smoothing_function=_SMOOTH)
    )


def _best_length_matched_chunks(
    pred_tokens: List[str],
    ref_tokens: List[str],
    max_stride: int = 256,
) -> Tuple[List[str], List[str]]:
    """
    Return (hyp_chunk, ref_chunk) of equal length with maximal smoothed BLEU4.

    If lengths already match, returns (pred, ref). Otherwise slides a window
    along the longer token sequence (stride-budgeted, then locally refined).
    """
    lp, lr = len(pred_tokens), len(ref_tokens)
    if lp == 0 or lr == 0:
        return pred_tokens, ref_tokens
    if lp == lr:
        return pred_tokens, ref_tokens

    def score_window(hyp: List[str], refw: List[str]) -> float:
        return _sentence_bleu(refw, hyp, _WEIGHTS4)

    if lp > lr:
        win = lr
        best_s, best_sc = 0, -1.0
        n = lp - win + 1
        step = max(1, n // max_stride)
        for s in range(0, n, step):
            sc = score_window(pred_tokens[s : s + win], ref_tokens)
            if sc > best_sc:
                best_sc, best_s = sc, s
        lo = max(0, best_s - step)
        hi = min(n, best_s + step + 1)
        for s in range(lo, hi):
            sc = score_window(pred_tokens[s : s + win], ref_tokens)
            if sc > best_sc:
                best_sc, best_s = sc, s
        return pred_tokens[best_s : best_s + win], ref_tokens

    # lp < lr : slide along reference
    win = lp
    best_s, best_sc = 0, -1.0
    n = lr - win + 1
    step = max(1, n // max_stride)
    for s in range(0, n, step):
        sc = score_window(pred_tokens, ref_tokens[s : s + win])
        if sc > best_sc:
            best_sc, best_s = sc, s
    lo = max(0, best_s - step)
    hi = min(n, best_s + step + 1)
    for s in range(lo, hi):
        sc = score_window(pred_tokens, ref_tokens[s : s + win])
        if sc > best_sc:
            best_sc, best_s = sc, s
    return pred_tokens, ref_tokens[best_s : best_s + win]


def cal_bleu_rouge(pred: str, ref: str) -> dict:
    """
    Smoothed sentence BLEU and ROUGE-L.

    BLEU is computed on length-matched contiguous token spans when pred and ref
    lengths disagree (see module docstring).
    """
    pred_tokens = safe_tokenize(pred)
    ref_tokens = safe_tokenize(ref)
    hyp_c, ref_c = _best_length_matched_chunks(pred_tokens, ref_tokens)

    bleu = _sentence_bleu(ref_c, hyp_c, _WEIGHTS4)
    bleu1 = _sentence_bleu(ref_c, hyp_c, (1, 0, 0, 0))
    bleu2 = _sentence_bleu(ref_c, hyp_c, (0, 1, 0, 0))
    bleu3 = _sentence_bleu(ref_c, hyp_c, (0, 0, 1, 0))
    bleu4 = _sentence_bleu(ref_c, hyp_c, (0, 0, 0, 1))

    hyp_joined = " ".join(hyp_c)
    ref_joined = " ".join(ref_c)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = scorer.score(hyp_joined, ref_joined)["rougeL"].fmeasure

    return {
        "bleu": bleu,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "rougeL": float(rougeL),
    }


if __name__ == "__main__":
    # Sanity: long repetitive pred vs short ref should not underflow to 0.
    ref = "the cat sat on the mat ."
    bad = "foo bar " * 200 + ref + " baz " * 200
    s = cal_bleu_rouge(bad, ref)
    assert s["bleu"] > 0.01, s
    assert s["bleu4"] > 0.0
    z = cal_bleu_rouge("", ref)
    assert z["bleu"] == 0.0
    print("ok", {k: round(v, 4) for k, v in s.items() if k.startswith("bleu")})
