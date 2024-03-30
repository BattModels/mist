import fileinput
import json

vocab = set()

# Add Atomic Symbols
for sym in fileinput.input():
    vocab.add(sym.strip())

# Add Bonds
vocab.update([".", "-", "=", "#", "$", ":", "/", "\\"])

# Structure
vocab.update(["%", "(", ")", "/", "\\", "@", "@@", "Branch", "Ring", "_"])

# Brackets and numbers
vocab.update(["[", "]", "Expl", "+", "-", *(str(x) for x in range(9))])

# Special tokens
vocab.update(["[UNK]", "[PAD]", "[MASK]", "[CLS]", "[SEP]", "[BOS]", "[EOS]"])

vocab = list(vocab)
vocab.sort()
print(json.dumps({v: k for k, v in enumerate(vocab)}))