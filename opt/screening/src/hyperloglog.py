import hashlib
import math
from typing import Callable


class HyperLogLogSet:
    def __init__(
        self,
        b: int = 12,
        hash: Callable = hashlib.sha1,
    ):
        if not (4 <= b <= 16):
            raise ValueError("Precision 'b' must be between 4 and 16.")
        self.b = b
        self.m = 1 << b
        self.hash = hash
        self.registers = [0] * self.m
        self.alpha = self._get_alpha()
        self.digest_bits = self.hash().digest_size * 8

    def insert(self, item: str):
        digest_bytes = self.hash(item.encode("utf-8")).digest()
        digest = int.from_bytes(digest_bytes, "big")  # fixed-size int

        idx = digest >> (self.digest_bits - self.b)
        w = digest & ((1 << (self.digest_bits - self.b)) - 1)
        rank = self._rho(w, self.digest_bits - self.b)
        self.registers[idx] = max(self.registers[idx], rank)

    def extend(self, items):
        for item in items:
            self.insert(item)

    def __len__(self):
        indicator = sum(2.0**-reg for reg in self.registers)
        raw_estimate = self.alpha * self.m**2 / indicator

        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            V = self.registers.count(0)
            if V > 0:
                return int(self.m * math.log(self.m / V))

        # Large range correction (not often needed)
        if raw_estimate > (1 << 32) / 30:
            return int(-(1 << 32) * math.log(1 - raw_estimate / (1 << 32)))

        return int(raw_estimate)

    def merge(self, other: "HyperLogLogSet"):
        if self.b != other.b:
            raise ValueError("Cannot merge HLL sets with different precision values")
        self.registers = [
            max(r1, r2) for r1, r2 in zip(self.registers, other.registers)
        ]

    def reduce(self, fabric) -> "HyperLogLogSet":
        """
        Perform an all-reduce using max() to merge HLL registers across processes.
        Returns a new HyperLogLogSet instance.
        """
        registers = fabric.all_reduce(self.registers.copy(), reduce_op="max")
        hll = HyperLogLogSet(self.b, hash=self.hash)
        hll.registers = registers
        return hll

    def _rho(self, w: int, max_bits: int) -> int:
        if w == 0:
            return max_bits + 1
        return max_bits - w.bit_length() + 1

    def _get_alpha(self) -> float:
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / self.m)


if __name__ == "__main__":
    import sys

    hll = HyperLogLogSet()
    for line in sys.stdin:
        hll.insert(line.strip())

    print(len(hll))
