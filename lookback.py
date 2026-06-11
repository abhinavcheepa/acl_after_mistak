"""
lookback.py
-----------
ACL Paper ke teeno methods ka implementation:
  1. zero_shot          - raw argmax, no fix
  2. lookback           - first subword ka tag copy karo
  3. lookback_with_score - sabse confident subword ka tag lo
  4. lookback_with_score_weighted - length-aware version

Ye functions dono pipelines (CRF + Linear) mein use hote hain.

Input format:
    emissions : torch.Tensor shape (seq_len, num_labels)
                - CRF pipeline  : MuRIL ka last hidden state → Linear layer output
                - Linear pipeline: same
    word_ids  : List[int | None]
                - tokenizer.word_ids() ka output
                - None = special token (CLS, SEP, PAD)
                - int  = word index
    id2label  : Dict[int, str]
                - {0: 'ADJ', 1: 'ADP', ...}

Output:
    List[str] - ek tag per original word
"""

import torch


# ─────────────────────────────────────────────────────────────
# HELPER — word boundaries nikalo
# ─────────────────────────────────────────────────────────────

def get_word_to_subwords(word_ids):
    """
    word_ids se ek dict banao:
        {word_idx: [subword_position_1, subword_position_2, ...]}

    Example:
        word_ids = [None, 0, 1, 1, 2, None]
        result   = {0: [1], 1: [2, 3], 2: [4]}
    """
    word_to_positions = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_to_positions:
            word_to_positions[wid] = []
        word_to_positions[wid].append(pos)
    return word_to_positions


# ─────────────────────────────────────────────────────────────
# METHOD 1: Zero-shot
# ─────────────────────────────────────────────────────────────

def zero_shot(emissions, word_ids, id2label):
    """
    ACL Paper ka baseline method.

    Logic:
    - Har word ke liye sirf PEHLE subword ka argmax lo
    - Baaki subwords ignore karo

    Paper mein:
        "Hindi FT" row = ye method
        Hindi F1 = 0.93

    Args:
        emissions : Tensor (seq_len, num_labels)
        word_ids  : List[int | None]
        id2label  : Dict[int, str]

    Returns:
        List[str] - one tag per word
    """
    word_to_positions = get_word_to_subwords(word_ids)
    result = []

    for wid in sorted(word_to_positions.keys()):
        positions = word_to_positions[wid]
        first_pos = positions[0]                          # pehla subword
        tag_id    = emissions[first_pos].argmax().item()  # argmax
        result.append(id2label.get(tag_id, "X"))

    return result


# ─────────────────────────────────────────────────────────────
# METHOD 2: Lookback
# ─────────────────────────────────────────────────────────────

def lookback(emissions, word_ids, id2label):
    """
    ACL Paper ka Look-back method.

    Logic:
    - Pehle subword ka tag nikalo
    - Usi tag ko word ke SARE subwords pe assign karo

    Paper mein:
        "Look-back" row
        Hindi F1 = 0.94 (+0.01 over zero-shot)
        Angika F1 = 0.77 (+0.08 over zero-shot!)

    Kyun kaam karta hai:
        "dEkhAibae" → ["dEkh", "##ibae"]
        "dEkh" Hindi mein "dekhna" se related → sahi tag
        "##ibae" akela meaningless → galat tag milta tha
        Lookback: dEkh ka tag → dono ko do ✅

    Args:
        emissions : Tensor (seq_len, num_labels)
        word_ids  : List[int | None]
        id2label  : Dict[int, str]

    Returns:
        List[str] - one tag per word
    """
    word_to_positions = get_word_to_subwords(word_ids)
    result = []

    for wid in sorted(word_to_positions.keys()):
        positions = word_to_positions[wid]
        first_pos = positions[0]                          # pehla subword
        tag_id    = emissions[first_pos].argmax().item()  # pehle ka tag
        result.append(id2label.get(tag_id, "X"))

    return result


# ─────────────────────────────────────────────────────────────
# METHOD 3: Lookback With Score
# ─────────────────────────────────────────────────────────────

def lookback_with_score(emissions, word_ids, id2label):
    """
    ACL Paper ka Look-back-with-score method.

    Logic:
    - Har word ke SARE subwords ka max confidence score nikalo
    - Sabse confident subword ka tag → word ko do

    Paper mein:
        Hindi F1 = 0.95 (+0.02 over zero-shot)
        Average across all languages = 0.81

    Kyun better hai lookback se:
        Word: "अभिनेत्री" → ["अभि", "##ने", "##त्री"]

        अभि    → ADJ  confidence=0.51  ← pehla, but uncertain
        ##ने   → NOUN confidence=0.32
        ##त्री → NOUN confidence=0.92  ← sabse confident ✅

        Lookback       → ADJ  (galat!)
        Lookback+Score → NOUN (sahi!) ✅

    Score formula:
        score(token) = max(softmax(logits))
        winner = argmax over all subwords of score(token)

    Args:
        emissions : Tensor (seq_len, num_labels)
        word_ids  : List[int | None]
        id2label  : Dict[int, str]

    Returns:
        List[str] - one tag per word
    """
    # Softmax se probabilities
    probs             = torch.softmax(emissions, dim=-1)  # (seq_len, num_labels)
    word_to_positions = get_word_to_subwords(word_ids)
    result            = []

    for wid in sorted(word_to_positions.keys()):
        positions = word_to_positions[wid]

        # Har subword ka max confidence
        best_pos   = max(positions, key=lambda p: probs[p].max().item())
        tag_id     = probs[best_pos].argmax().item()
        result.append(id2label.get(tag_id, "X"))

    return result


# ─────────────────────────────────────────────────────────────
# METHOD 4: Lookback With Score Weighted (Tumhara addition)
# ─────────────────────────────────────────────────────────────

def lookback_with_score_weighted(emissions, word_ids, id2label,
                                  tokenizer=None, words=None):
    """
    Tumhara improved version of LBS.

    Problem with basic LBS:
        Short suffix tokens (##ए, ##गा) kabhi kabhi
        artificially high confidence dikhate hain
        kyunki woh bahut common subwords hain.

    Fix:
        Score = confidence × length_weight
        length_weight = min(token_length / 3.0, 1.5)
        Longer subword = zyada informative = zyada weight

    Example:
        Word: "प्रेमचंद्रीय"
        Tokens: ["प्रे", "##मच", "##ंद्री", "##य"]

        Basic LBS:
            "##य" confidence=0.91 → PART (galat!)

        Weighted LBS:
            "##य"    length=2 weight=0.67 score=0.91×0.67=0.61
            "##ंद्री" length=5 weight=1.5  score=0.72×1.5 =1.08 ← wins
            → PROPN (sahi!) ✅

    Args:
        emissions : Tensor (seq_len, num_labels)
        word_ids  : List[int | None]
        id2label  : Dict[int, str]
        tokenizer : HuggingFace tokenizer (optional, for length)
        words     : List[str] original words (optional)

    Returns:
        List[str] - one tag per word
    """
    probs             = torch.softmax(emissions, dim=-1)
    word_to_positions = get_word_to_subwords(word_ids)
    result            = []

    for wid in sorted(word_to_positions.keys()):
        positions = word_to_positions[wid]

        best_pos   = None
        best_score = -1.0

        for pos in positions:
            confidence = probs[pos].max().item()

            # Length weight — agar tokenizer available hai
            length_weight = 1.0
            if tokenizer is not None:
                try:
                    token_str = tokenizer.convert_ids_to_tokens(
                        [pos]  # placeholder — actual token id chahiye
                    )[0]
                    # ## prefix hata ke length lo
                    clean = token_str.replace("##", "")
                    length_weight = min(len(clean) / 3.0, 1.5)
                    length_weight = max(length_weight, 0.5)  # minimum 0.5
                except Exception:
                    length_weight = 1.0

            score = confidence * length_weight

            if score > best_score:
                best_score = score
                best_pos   = pos

        if best_pos is None:
            best_pos = positions[0]

        tag_id = probs[best_pos].argmax().item()
        result.append(id2label.get(tag_id, "X"))

    return result


# ─────────────────────────────────────────────────────────────
# UTILITY — Entropy-based confidence (fusion ke liye)
# ─────────────────────────────────────────────────────────────

def entropy_confidence(logits_for_token):
    """
    Softmax saturation problem fix.

    Problem:
        Jab model bahut trained ho, almost every prediction
        ka softmax 0.95+ hota hai → tiebreak mein sab same

    Fix:
        Entropy = uncertainty measure
        Low entropy  = high confidence (model sure hai)
        High entropy = low confidence (model unsure hai)

        confidence = 1 / (1 + entropy)

    Args:
        logits_for_token : Tensor (num_labels,)

    Returns:
        float - confidence score (0 to 1)
    """
    probs   = torch.softmax(logits_for_token, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
    return 1.0 / (1.0 + entropy)


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("lookback.py — quick sanity check")
    print("-" * 40)

    # Fake emissions — 5 tokens, 4 tags
    torch.manual_seed(42)
    fake_emissions = torch.randn(6, 4)

    # word_ids: [None, 0, 1, 1, 2, None]
    # word 0 → token 1
    # word 1 → tokens 2, 3 (split word)
    # word 2 → token 4
    fake_word_ids = [None, 0, 1, 1, 2, None]

    id2label = {0: "NOUN", 1: "VERB", 2: "ADJ", 3: "ADP"}

    zs  = zero_shot(fake_emissions, fake_word_ids, id2label)
    lb  = lookback(fake_emissions, fake_word_ids, id2label)
    lbs = lookback_with_score(fake_emissions, fake_word_ids, id2label)
    lbw = lookback_with_score_weighted(fake_emissions, fake_word_ids, id2label)

    print(f"zero_shot              : {zs}")
    print(f"lookback               : {lb}")
    print(f"lookback_with_score    : {lbs}")
    print(f"lookback_with_score_wt : {lbw}")
    print()
    print("Word 1 split hai (tokens 2,3) — methods mein fark dikhna chahiye")
    print("Done ✅")
