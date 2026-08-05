# trigram

A character-level **trigram** language model that generates names, built as a
modification of the bigram model from Andrej Karpathy's
[Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
(makemore, part 1).

Like the original it is a pure counting model — no neural network, no gradients.
It counts how often each character follows a given context in a corpus of 32,033
names, normalizes those counts into probability distributions, and samples from
them.

The difference is how much context it conditions on. Karpathy's bigram model
predicts the next character from **one** preceding character. This one predicts
from **two**.

```
ce.
bra.
jalius.
rochityharlonimittain.
luwak.
```

## Modifications from the bigram original

### 1. Two characters of context instead of one

The bigram model stores a 27×27 matrix of `P(next | prev)` — 27 being the 26
letters plus `.` as a start/end marker. This model stores a **729×27** matrix of
`P(next | prev1, prev2)`: one row for each of the 27×27 possible two-character
contexts, one column for each possible next character.

More context means sharper predictions. The bigram model only knows "the last
letter was `n`"; this one knows "the last two letters were `a`, `n`", which is a
far better predictor of what comes next.

### 2. Contexts are encoded arithmetically, not with a lookup table

The obvious way to index 729 contexts is a dictionary mapping `'an'` → row number.
That turns out to be awkward: it creates a *second* numbering system alongside
`char_to_int`, and the two disagree about which integer means what.

Instead a context is treated as a **two-digit number in base 27**:

```python
row = char_to_int[ch1] * 27 + char_to_int[ch2]
```

This agrees with `char_to_int` by construction, needs no extra dictionary, and
makes the sampling step fall out as one line of arithmetic (below). The inverse,
used for plot labels, is `int_to_char[i // 27] + int_to_char[i % 27]`.

### 3. Sampling slides the context with modular arithmetic

The bigram sampler can feed each sampled character straight back in as the next
index, because context and prediction share one 27-symbol alphabet. A trigram
sampler cannot — the row index lives in 0–728 and the sampled value in 0–26, so
they need translating between.

Rather than tracking the context as a string and re-encoding it every step, the
window slides arithmetically:

```python
ix = (ix % 27) * 27 + sample
```

`ix % 27` recovers the *second* character of the current context (the low base-27
digit); it becomes the first character of the new context, with the freshly
sampled character as the new second. Same sliding window, no string handling.

### 4. Padding is two leading markers, not one

The bigram model pads each word as `.` + word + `.`. A trigram model needs
`.` `.` + word + `.` — two leading markers, so that the *first* real character has
a full two-character context to be predicted from. Without it there is no row
representing "nothing has been generated yet" and sampling has nowhere to start.

### 5. Different visualization strategy

The bigram model's 27×27 matrix fits in a single annotated heatmap — 729 cells,
each labeled with its character pair and count. That does not survive the jump to
729×27, which is 19,683 cells and roughly 40,000 text objects.

Two views replace it:

- **`plot_overview(N)`** — the whole matrix, no per-cell text. Uses
  `aspect='auto'` so a 729×27 matrix fits a normal figure instead of rendering as
  an unreadable needle, and a **logarithmic** color scale because the counts span
  0 to ~6,800 and a linear ramp leaves everything but a handful of cells
  indistinguishably pale. Contexts that never occur are flat gray, so the sparsity
  pattern reads directly.
- **`plot_pair_block(N, 'a')`** — the 27 contexts beginning with a given
  character. Small enough that every cell can carry its count. Pass `'.'` to see
  the word-start distribution.

## Running it

Requires `torch` and `matplotlib`:

```bash
python makemore_pt1.py
```

Note that `plot_overview` and `plot_pair_block` are called at module level, so two
plot windows open and the script waits for you to close them before printing the
generated names. Comment out those two calls to skip straight to sampling, or pass
`save_path='foo.png'` to write a PNG instead of opening a window.

The sampler is seeded, so the five names it prints are the same on every run.

## Checking correctness

Two invariants that catch structural errors far more reliably than reading the
generated names:

- `N.sum()` should equal `sum(len(w) + 1 for w in words)` — 228,146 for this
  corpus. Any mismatch means the padding or the sliding window is off.
- Individual cells should agree with the raw text. For example the number of names
  ending in `na` is 1,673, which is exactly `N[char_to_int['n'] * 27 + char_to_int['a'], 0]`.
  This check is worth doing on an asymmetric trigram, because a transposed matrix
  passes the sum check but fails this one.

## Known limitations

- **No smoothing.** Contexts that never appear in the data have a row sum of zero,
  so the corresponding rows of `P` are `nan`. Sampling never reaches them — you
  can only arrive at a context you have seen — but any likelihood evaluation would
  need add-one smoothing first.
- **No loss computed.** The natural next step is average negative log-likelihood,
  which is what actually demonstrates the trigram model beats the bigram one.
- `makemore_pt1_bigram.ipynb` is the original bigram work this was built from.

## Credit

All of the underlying ideas, and the `names.txt` dataset, come from Andrej
Karpathy's [makemore](https://github.com/karpathy/makemore) and the
[Zero to Hero](https://karpathy.ai/zero-to-hero.html) lecture series.
