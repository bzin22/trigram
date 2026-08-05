# trigram

A character-level trigram model that generates names. It is a modification of the
bigram model from Andrej Karpathy's
[Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
(makemore, part 1).

It counts how often each character follows a given context across 32,033 names,
normalizes the counts into probability distributions, and samples from them. No
neural network, no gradients.

Karpathy's bigram model predicts the next character from one preceding character.
This one uses two.

```
ce.
bra.
jalius.
rochityharlonimittain.
luwak.
```

## Modifications from the bigram original

### Two characters of context instead of one

The bigram model stores a 27x27 matrix of `P(next | prev)`, with 27 being the 26
letters plus `.` as a start/end marker. This one stores a 729x27 matrix of
`P(next | prev1, prev2)`: a row for each of the 27x27 two-character contexts, a
column for each next character.

### Contexts encoded as base-27 numbers

Indexing 729 contexts with a dictionary from `'an'` to a row number creates a
second numbering system alongside `char_to_int`, and the two disagree about which
integer means what. Instead a context is a two-digit number in base 27:

```python
row = char_to_int[ch1] * 27 + char_to_int[ch2]
```

That agrees with `char_to_int` by construction and needs no extra dictionary. The
inverse, used for plot labels, is `int_to_char[i // 27] + int_to_char[i % 27]`.

### Sampling slides the context with modular arithmetic

A bigram sampler can feed each sampled character straight back in, because context
and prediction share one 27-symbol alphabet. A trigram sampler cannot. The row
index lives in 0-728 and the sampled value in 0-26.

Rather than track the context as a string and re-encode it every step, the window
slides arithmetically:

```python
ix = (ix % 27) * 27 + sample
```

`ix % 27` recovers the second character of the current context (the low base-27
digit). It becomes the first character of the new context, with the sampled
character as the new second.

### Two leading markers in the padding

The bigram model pads each word as `.` + word + `.`. A trigram model needs
`.` `.` + word + `.` so the first real character has a full two-character context.
Without it there is no row for "nothing generated yet" and sampling cannot start.

### Different visualization

The bigram model's 27x27 matrix fits in one annotated heatmap, 729 cells each
labeled with its pair and count. That does not survive the jump to 729x27, which
is 19,683 cells and about 40,000 text objects.

Two views replace it:

- `plot_overview(N)` renders the whole matrix with no per-cell text. It uses
  `aspect='auto'` so a 729x27 matrix fits a normal figure instead of a needle, and
  a log color scale because counts run from 0 to about 6,800. A linear ramp leaves
  everything but a few cells indistinguishably pale. Contexts that never occur are
  flat gray, so the sparsity pattern reads directly.
- `plot_pair_block(N, 'a')` renders the 27 contexts starting with a given
  character, small enough to label every cell with its count. Pass `'.'` for the
  word-start distribution.

## Running it

Needs `torch` and `matplotlib`.

```bash
python makemore_pt1.py
```

`plot_overview` and `plot_pair_block` are called at module level, so two plot
windows open and the script waits for you to close them before printing names.
Comment out those calls to skip to sampling, or pass `save_path='foo.png'` to
write a PNG instead of opening a window.

The sampler is seeded, so the five names are the same every run.

## Checking correctness

Two invariants that catch structural errors better than reading the output:

- `N.sum()` should equal `sum(len(w) + 1 for w in words)`, which is 228,146 here.
  A mismatch means the padding or the sliding window is off.
- Individual cells should match the raw text. 1,673 names end in `na`, which is
  exactly `N[char_to_int['n'] * 27 + char_to_int['a'], 0]`. Use an asymmetric
  trigram for this check. A transposed matrix passes the sum check but fails this
  one.

## Known limitations

- No smoothing. Contexts absent from the data have a row sum of zero, so those
  rows of `P` are `nan`. Sampling never reaches them, since you can only arrive at
  a context you have seen, but likelihood evaluation would need add-one smoothing
  first.
- No loss computed. Average negative log-likelihood is the next step, and it is
  what actually shows the trigram beats the bigram.
- `makemore_pt1_bigram.ipynb` is the original bigram work this was built from.

## Credit

The ideas and the `names.txt` dataset come from Andrej Karpathy's
[makemore](https://github.com/karpathy/makemore) and the
[Zero to Hero](https://karpathy.ai/zero-to-hero.html) lectures.
