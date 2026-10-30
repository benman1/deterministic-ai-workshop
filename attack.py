"""A TextFooler-style word-substitution attack, written from the paper's Algorithm 1.

Jin, Jin, Zhou and Szolovits, "Is BERT Really Robust?", AAAI 2020 (https://arxiv.org/abs/1907.11932).
This is not the authors' code. It follows their algorithm box and settings (N = 50 neighbors,
word-vector cosine above 0.7, same part of speech, stop words skipped) with three substitutions,
each named in notebook 09: sentence similarity comes from all-MiniLM-L6-v2 instead of the
Universal Sentence Encoder, the part of speech comes from NLTK's perceptron tagger, and the
attacker's goal can be "act on a wrong route" as well as "flip the label".
"""
from __future__ import annotations

import re

import numpy as np

TOKEN = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|[^A-Za-z]+")


def split(text: str) -> list[str]:
    """Words and the gaps between them, so ''.join(split(t)) == t."""
    return TOKEN.findall(text)


def is_word(tok: str) -> bool:
    return tok[0].isalpha()


def coarse(tag: str) -> str:
    return {"NN": "N", "VB": "V", "JJ": "J", "RB": "R"}.get(tag[:2], tag)


def nltk_pos(words: list[str]) -> list[str]:
    import nltk

    return [coarse(t) for _, t in nltk.pos_tag(words)]


def stop_words() -> set[str]:
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


def load_vectors(path, vocab_limit: int | None = None):
    """Counter-fitted vectors (Mrksic et al. 2016): one word and 300 floats per line."""
    words, rows = [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if not parts[0].isalpha():
                continue
            words.append(parts[0])
            rows.append(np.asarray(parts[1:], dtype=np.float32))
            if vocab_limit and len(words) >= vocab_limit:
                break
    mat = np.stack(rows)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return words, mat


class Attacker:
    """victim(texts) -> (n, classes) probabilities; sim(a, list_of_b) -> cosine scores."""

    def __init__(self, words, mat, victim, sim, pos=nltk_pos, stop=None, n_syn=50, delta=0.7, eps=0.8, anchor_original=False):
        self.words, self.mat, self.victim, self.sim, self.pos = words, mat, victim, sim, pos
        self.index = {w: i for i, w in enumerate(words)}
        self.stop = stop if stop is not None else stop_words()
        self.n_syn, self.delta, self.eps, self.anchor = n_syn, delta, eps, anchor_original
        self._syn_cache: dict[str, list[str]] = {}

    def synonyms(self, word: str) -> list[str]:
        if word in self._syn_cache:
            return self._syn_cache[word]
        i = self.index.get(word)
        out: list[str] = []
        if i is not None:
            cos = self.mat @ self.mat[i]
            for j in np.argsort(-cos)[: self.n_syn + 1]:
                if j != i and cos[j] > self.delta:
                    out.append(self.words[j])
        self._syn_cache[word] = out
        return out

    def _swap(self, toks: list[str], i: int, cand: str) -> list[str]:
        new = cand.capitalize() if toks[i][0].isupper() else cand
        return toks[:i] + [new] + toks[i + 1 :]

    def _same_pos(self, toks: list[str], i: int, trials: list[list[str]]) -> list[bool]:
        pos_i = [k for k, t in enumerate(toks) if is_word(t)].index(i)
        before = self.pos([t for t in toks if is_word(t)])[pos_i]
        return [self.pos([t for t in tr if is_word(t)])[pos_i] == before for tr in trials]

    def attack(self, text: str, true_idx: int, success) -> dict:
        """Greedy swaps until success(probs_row) is true. success sees one row of probabilities.

        first_flip records the message at the first step where the top label differs from true_idx,
        so one run answers both "flip the label" and "reach the success condition".
        """
        toks = split(text)
        p0 = self.victim([text])[0]
        queries = 1
        idx = [i for i, t in enumerate(toks) if is_word(t) and t.lower() not in self.stop and t.lower() in self.index]
        dels = ["".join(toks[:i] + toks[i + 1 :]) for i in idx]
        if dels:
            P = self.victim(dels)
            queries += len(dels)
            order = [idx[j] for j in np.argsort(-(p0[true_idx] - P[:, true_idx]))]
        else:
            order = []
        cur, cur_p, changes = list(toks), p0, []
        flip = None
        for i in order:
            syn = self.synonyms(toks[i].lower())
            if not syn:
                continue
            trials = [self._swap(cur, i, c) for c in syn]
            same = self._same_pos(cur, i, trials)
            trials = [t for t, ok in zip(trials, same) if ok]
            syn = [c for c, ok in zip(syn, same) if ok]
            if not trials:
                continue
            texts = ["".join(t) for t in trials]
            sims_cur = np.asarray(self.sim(text if self.anchor else "".join(cur), texts))
            keep = [k for k in range(len(texts)) if sims_cur[k] > self.eps]
            if not keep:
                continue
            P = self.victim([texts[k] for k in keep])
            queries += len(keep)
            win = [j for j, k in enumerate(keep) if success(P[j])]
            if win:
                sims_orig = np.asarray(self.sim(text, [texts[keep[j]] for j in win]))
                j = win[int(np.argmax(sims_orig))]
                cur, cur_p = trials[keep[j]], P[j]
                changes.append((toks[i], syn[keep[j]]))
                flip = flip or (cur, len(changes), queries)
                return self._result(text, cur, cur_p, True, queries, changes, toks, flip)
            j = int(np.argmin(P[:, true_idx]))
            if P[j, true_idx] < cur_p[true_idx]:
                cur, cur_p = trials[keep[j]], P[j]
                changes.append((toks[i], syn[keep[j]]))
                if flip is None and int(np.argmax(cur_p)) != true_idx:
                    flip = (cur, len(changes), queries)
        return self._result(text, cur, cur_p, False, queries, changes, toks, flip)

    def _result(self, text, cur, cur_p, ok, queries, changes, toks, flip) -> dict:
        adv = "".join(cur)
        return dict(adv=adv, success=bool(ok), queries=queries, changes=changes,
                    n_words=sum(is_word(t) for t in toks), n_changed=len(changes),
                    sim_to_original=float(self.sim(text, [adv])[0]), probs=[float(x) for x in cur_p],
                    flip=None if flip is None else dict(adv="".join(flip[0]), n_changed=flip[1], queries=flip[2],
                                                        sim_to_original=float(self.sim(text, ["".join(flip[0])])[0])))

    def random_control(self, text: str, k: int, rng: np.random.Generator, draws: int = 20) -> list[str]:
        """k random swaps of non-stop words for random same-POS neighbors; no victim queries."""
        toks = split(text)
        idx = [i for i, t in enumerate(toks) if is_word(t) and t.lower() not in self.stop and self.synonyms(t.lower())]
        out = []
        for _ in range(draws):
            cur = list(toks)
            for i in rng.permutation(idx)[: max(k, 1)]:
                syn = self.synonyms(toks[i].lower())
                trials = [self._swap(cur, i, c) for c in syn]
                ok = [t for t, same in zip(trials, self._same_pos(cur, i, trials)) if same]
                if ok:
                    cur = ok[int(rng.integers(len(ok)))]
            out.append("".join(cur))
        return out
