import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

words = open('/Users/bzin/nn-zero-to-hero/data/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))

# goal is to build a 27x27 array of all the combinations of 2 characters that you can make with the 26 letters of the alphabet
# within that array is the number of times that those pairs of characters show up in the names.txt file

char_to_int = {s:i+1 for i,s in enumerate(chars)}
char_to_int['.'] = 0
# char_to_int
# {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,'i': 9, 'j': 10,
#  'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
#  't': 20, 'u': 21, 'v': 22, 'w':23, 'x': 24, 'y': 25, 'z': 26, '.': 0}
int_to_char = {i:s for s,i in char_to_int.items()}
# int_to_char
# {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h',9: 'i', 10: 'j', 
# 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 
# 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '.'}

new_chars = chars.copy()
new_chars.append('.')

new_char_to_int = {}
count = 1
for c in new_chars:
    for d in new_chars:
        new_char_to_int[c+d] = count
        count += 1
new_char_to_int['..'] = 0 

N = torch.zeros((729,27), dtype=torch.float32)
# create the probability weight matrix for each letter pair based on the input data
for w in words: 
    chs = ['.', '.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = char_to_int[ch1] * N.shape[1] + char_to_int[ch2]
        ix2= char_to_int[ch3]
        N[ix1, ix2] += 1


new_int_to_char = {i:s for s,i in new_char_to_int.items()}
# print(sum(len(w)+1 for w in words))
# print(new_char_to_int)
# print("N shape: ", N.shape)

# Visualize the weight matrix of contexts and their frequency.
# N is 729x27, so 20k cells -- far too many to annotate. Two views instead:
# an overview showing where the counts live, and a drill-in on one block.

# counts span 0 to ~6800, so a linear color ramp leaves everything but a few
# cells indistinguishably pale. Log scale, with never-seen (0) called out flat.
_cmap = plt.get_cmap('Greens').copy()
_cmap.set_bad('#f2f2f2')
_cmap.set_under('#f2f2f2')

def _decode_context(i):
    """Row index -> the two-character context it stands for (inverse of ch1*27+ch2)."""
    return int_to_char[i // 27] + int_to_char[i % 27]

def plot_overview(N, save_path=None):
    """All 729 context-rows at once. No cell text -- read this for structure."""
    norm = LogNorm(vmin=1, vmax=N.max().item())
    fig, ax = plt.subplots(figsize=(5.5, 9))
    im = ax.imshow(N, cmap=_cmap, norm=norm, aspect='auto', interpolation='nearest')

    ax.set_xticks(range(27))
    ax.set_xticklabels([int_to_char[j] for j in range(27)], fontsize=7)
    # one y label per block of 27, i.e. each time the context's first character changes
    blocks = list(range(0, N.shape[0], 27))
    ax.set_yticks(blocks)
    ax.set_yticklabels([_decode_context(i) for i in blocks], fontsize=7)

    ax.set_xlabel('next character (ch3)', fontsize=9)
    ax.set_ylabel('context (ch1, ch2)', fontsize=9)
    ax.set_title('Trigram counts (log color scale)', fontsize=11)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    cb = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
    cb.set_label('count', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    _finish(fig, save_path)

def plot_pair_block(N, first_char, save_path=None):
    """The 27 contexts starting with `first_char` -- small enough to label."""
    base = char_to_int[first_char] * N.shape[1]
    sub = N[base:base + 27]
    norm = LogNorm(vmin=1, vmax=max(sub.max().item(), 1))

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(sub, cmap=_cmap, norm=norm, interpolation='nearest')

    ax.set_xticks(range(27))
    ax.set_xticklabels([int_to_char[j] for j in range(27)], fontsize=9)
    ax.set_yticks(range(27))
    ax.set_yticklabels([first_char + int_to_char[i] for i in range(27)], fontsize=9)

    for i in range(27):
        for j in range(27):
            v = sub[i, j].item()
            if v == 0:
                continue
            # dark cells need light ink to stay legible
            color = 'white' if norm(v) > 0.65 else '#333333'
            ax.text(j, i, int(v), ha='center', va='center', fontsize=7, color=color)

    ax.set_xlabel('next character (ch3)', fontsize=10)
    ax.set_ylabel(f"context (ch1, ch2) starting '{first_char}'", fontsize=10)
    ax.set_title(f"Trigram counts for contexts '{first_char}.' through '{first_char}z'", fontsize=12)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    _finish(fig, save_path)

def _finish(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

plot_overview(N)
plot_pair_block(N, 'a')

P = N / torch.sum(N, 1, keepdim=True) # normalize the weights in N to prepare for probability distribution
# print(P[0])

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True: 
        
        p = P[ix] # starts at '..'
        # print("before: ", ix)
        sample = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # picks the next likely character based on the input prob distribution
        # print("after: ", ix)
        ix = (ix % 27) * 27 + sample
        out.append(int_to_char[sample]) # adds that character to the output
        if sample == 0: # if a '.' is seen again, word has ended
            break
    print(''.join(out))
