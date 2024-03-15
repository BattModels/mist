import fileinput
import json

vocab = set()

# Add Atomic Symbols
for sym in fileinput.input():
    vocab.add(sym.strip())

# Add Aromatics
vocab.update(["b", "c", "n", "o", "p", "s"])

# Add Bonds
vocab.update([".", "-", "=", "#", "$", ":", "/", "\\"])

# Structure
vocab.update(["%", "(", ")", "/", "\\", "@", "@@"])

# Brackets and numbers
vocab.update(["[", "]", "+", "-", *(str(x) for x in range(9))])

# Special tokens
vocab.update(["[UNK]", "[PAD]", "[MASK]", "[CLS]", "[SEP]", "[BOS]", "[EOS]"])

vocab = list(vocab)
vocab.sort()
print(json.dumps({v: k for k, v in enumerate(vocab)}))
