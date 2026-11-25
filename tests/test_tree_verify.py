"""Compare the GPU tensor tree verify with the CPU reference on small trees."""
import sys
import torch

sys.path.insert(0, sys.argv[1])
from aibook_loader import verify_tree_greedy_musa, verify_tree_greedy_musa_torch


def run(cand, nxt, sib, tp, ri=None, bs=1):
    cand = torch.tensor([cand], dtype=torch.int32)
    nxt = torch.tensor([nxt], dtype=torch.int32)
    sib = torch.tensor([sib], dtype=torch.int32)
    tp = torch.tensor([tp], dtype=torch.int64)
    ri = torch.tensor([ri], dtype=torch.int32)
    n = cand.shape[1]
    def fresh():
        return (torch.zeros((bs, n), dtype=torch.int32),
                torch.full((bs, n), -1, dtype=torch.int32),
                torch.zeros((bs,), dtype=torch.int32))
    p1, a1, c1 = fresh()
    p2, a2, c2 = fresh()
    verify_tree_greedy_musa(p1, a1, c1, cand, ri, nxt, sib, tp, -1)
    verify_tree_greedy_musa_torch(p2, a2, c2, cand, ri, nxt, sib, tp, -1)
    assert torch.equal(p1, p2), (p1, p2)
    assert torch.equal(a1, a2), (a1, a2)
    assert torch.equal(c1, c2), (c1, c2)
    return int(c2[0]), a2.tolist()


def main():
    n = 6
    ri = list(range(n))
    cases = [
        ("copy_chain", [84799, 60, 198, 4848, 28, 0], [1, 2, 3, 4, -1, -1],
         [-1, 5, -1, -1, -1, -1], [60, 198, 4848, 28, 71061, 198], 4),
        ("full_chain", [100, 101, 102, 103, -1, -1], [1, 2, 3, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1], [101, 102, 103, 999, 0, 0], 3),
        ("branch_pick_sibling", [999, 111, 222, 333, 0, 0], [1, -1, 3, -1, -1, -1],
         [-1, 2, -1, -1, -1, -1], [222, 0, 333, 444, 0, 0], 2),
        ("reject_first", [999, 111, 0, 0, 0, 0], [1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1], [777, 0, 0, 0, 0, 0], 0),
    ]
    for name, cand, nxt, sib, tp, expected in cases:
        got, acc = run(cand, nxt, sib, tp, ri)
        print(name, "accept", got, "expected", expected, "acc", acc)
        assert got == expected, name
    print("PASS")


if __name__ == "__main__":
    main()
