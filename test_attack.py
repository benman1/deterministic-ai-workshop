import numpy as np

import attack as a


def toy_vectors():
    # unit vectors: "cancel" ~ "terminate" (0.8), "cancel" ~ "cancellation" (0.9, other part of speech), "the" ~ "that"
    words = ["cancel", "terminate", "cancellation", "booking", "reservation", "the", "that", "please"]
    v = {"cancel": [1, 0, 0], "terminate": [0.8, 0.6, 0], "cancellation": [0.9, 0.0, 0.436],
         "booking": [0, 1, 0], "reservation": [0.05, 0.99, 0], "the": [0, 0, 1], "that": [0.1, 0, 0.99], "please": [-1, 0, 0]}
    mat = np.array([v[w] for w in words], dtype=np.float32)
    return words, mat / np.linalg.norm(mat, axis=1, keepdims=True)


TAGS = {"cancel": "V", "terminate": "V", "cancellation": "N", "booking": "N", "reservation": "N", "the": "D", "that": "D", "please": "R"}


def victim(texts):
    # class 0 only while the word "cancel" is present; otherwise class 1
    return np.array([[0.9, 0.1] if "cancel" in t.lower().split() else [0.1, 0.9] for t in texts])


def make(sim=lambda a_, bs: [1.0] * len(bs), v=victim):
    words, mat = toy_vectors()
    pos = lambda ws: [TAGS.get(w.lower(), "X") for w in ws]
    return a.Attacker(words, mat, v, sim, pos=pos, stop={"the", "that", "please"}, n_syn=5, delta=0.7, eps=0.5)


def test_split_round_trips():
    t = "Please cancel the booking, it's urgent!"
    assert "".join(a.split(t)) == t


def test_swaps_the_deciding_word_for_a_same_pos_neighbor():
    r = make().attack("Please cancel the booking", 0, lambda p: p.argmax() != 0)
    assert r["success"] and r["adv"] == "Please terminate the booking"
    assert r["n_changed"] == 1  # "cancellation" is closer (0.9) but a noun: the part-of-speech check removes it


def test_stop_words_are_never_swapped():
    r = make().attack("Please cancel the booking", 0, lambda p: p.argmax() != 0)
    assert all(old not in {"the", "Please", "please"} for old, _ in r["changes"])


def test_similarity_floor_blocks_the_swap():
    r = make(sim=lambda a_, bs: [0.1] * len(bs)).attack("Please cancel the booking", 0, lambda p: p.argmax() != 0)
    assert not r["success"] and r["adv"] == "Please cancel the booking"


def test_attack_fails_when_no_neighbor_flips_the_victim():
    def stubborn(texts):
        return np.array([[0.9, 0.1]] * len(texts))
    r = make(v=stubborn).attack("Please cancel the booking", 0, lambda p: p.argmax() != 0)
    assert not r["success"] and r["n_changed"] == 0


def test_random_control_changes_at_most_k_words_and_never_queries_the_victim():
    calls = []
    at = make(v=lambda t: calls.append(t) or victim(t))
    outs = at.random_control("Please cancel the booking", 1, np.random.default_rng(0), draws=5)
    assert not calls and all(sum(x != y for x, y in zip(o.split(), "Please cancel the booking".split())) <= 1 for o in outs)


def test_anchor_compares_candidates_with_the_original_not_the_current_text():
    # two swaps are needed: the victim flips only when both "cancel" and "booking" are gone
    def two_word_victim(texts):
        p = {2: 0.9, 1: 0.7, 0: 0.1}
        return np.array([[p[("cancel" in t.split()) + ("booking" in t.split())], 1 - p[("cancel" in t.split()) + ("booking" in t.split())]] for t in texts])
    seen = {}
    for anchor in (False, True):
        refs = []
        def sim(x, ys, refs=refs):
            refs.append(x)
            return [1.0] * len(ys)
        at = make(sim=sim, v=two_word_victim)
        at.anchor = anchor
        r = at.attack("Please cancel the booking", 0, lambda p: p.argmax() != 0)
        assert r["success"] and r["n_changed"] == 2
        seen[anchor] = refs
    assert set(seen[True][:-1]) == {"Please cancel the booking"}
    assert len(set(seen[False][:-1])) == 2


def test_first_flip_is_recorded_with_its_swap_count():
    r = make().attack("Please cancel the booking", 0, lambda p: p.argmax() != 0)
    assert r["flip"]["n_changed"] == 1 and r["flip"]["adv"] == r["adv"]
