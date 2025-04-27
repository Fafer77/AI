from typing import List, Tuple
import random
from functools import lru_cache
import time

# Constants for cell states
UNKNOWN = -1  # undecided
EMPTY = 0     # white
FILLED = 1    # black

@lru_cache(None)
def all_placements(length: int, blocks: Tuple[int, ...]) -> List[List[int]]:
    """
    Generate all possible line fillings of given length with the specified blocks.
    """
    blocks_list = list(blocks)
    if not blocks_list:
        return [[EMPTY] * length]
    total_blocks = sum(blocks_list)
    gaps = length - total_blocks - (len(blocks_list) - 1)
    placements: List[List[int]] = []

    def helper(offset: int, rem_blocks: List[int], current: List[int]):
        if not rem_blocks:
            tail = [EMPTY] * (length - len(current))
            placements.append(current + tail)
            return
        blk = rem_blocks[0]
        for pad in range(gaps - offset + 1):
            prefix = [EMPTY] * pad
            block_cells = [FILLED] * blk
            sep = [EMPTY] if len(rem_blocks) > 1 else []
            new_current = current + prefix + block_cells + sep
            helper(offset + pad, rem_blocks[1:], new_current)

    helper(0, blocks_list, [])
    return placements

@lru_cache(None)
def infer_line(length: int, blocks: Tuple[int, ...], current: Tuple[int, ...]) -> List[int]:
    """
    Given a partially known line (as tuple), infer cells that are forced (same in all valid placements).
    Returns a list of ints with UNKNOWN, EMPTY, or FILLED.
    """
    placements = all_placements(length, blocks)
    valid: List[List[int]] = []
    for p in placements:
        if all(c == UNKNOWN or c == p[i] for i, c in enumerate(current)):
            valid.append(p)
    if not valid:
        return list(current)
    inferred = list(current)
    for i in range(length):
        vals = {p[i] for p in valid}
        if len(vals) == 1:
            inferred[i] = vals.pop()
    return inferred


def initial_inference(x: int, y: int,
                      rows_spec: List[List[int]], cols_spec: List[List[int]]) -> List[List[int]]:
    """
    Preprocessing inference: full-line fill and overlap (iterative inference).
    Returns grid with UNKNOWN, EMPTY, or FILLED.
    """
    grid = [[UNKNOWN] * y for _ in range(x)]
    # full-line inference
    for i, spec in enumerate(rows_spec):
        if spec and sum(spec) + len(spec) - 1 == y:
            idx = 0
            for blk in spec:
                for j in range(idx, idx + blk):
                    grid[i][j] = FILLED
                if idx + blk < y:
                    grid[i][idx + blk] = EMPTY
                idx += blk + 1
    for j, spec in enumerate(cols_spec):
        if spec and sum(spec) + len(spec) - 1 == x:
            idx = 0
            for blk in spec:
                for i0 in range(idx, idx + blk):
                    grid[i0][j] = FILLED
                if idx + blk < x:
                    grid[idx + blk][j] = EMPTY
                idx += blk + 1
    # overlap inference
    changed = True
    while changed:
        changed = False
        # rows
        for i in range(x):
            current = tuple(grid[i])
            blocks = tuple(rows_spec[i])
            new_row = infer_line(y, blocks, current)
            for j in range(y):
                if grid[i][j] == UNKNOWN and new_row[j] != UNKNOWN:
                    grid[i][j] = new_row[j]
                    changed = True
        # cols
        for j in range(y):
            col = tuple(grid[i][j] for i in range(x))
            blocks = tuple(cols_spec[j])
            new_col = infer_line(x, blocks, col)
            for i0 in range(x):
                if grid[i0][j] == UNKNOWN and new_col[i0] != UNKNOWN:
                    grid[i0][j] = new_col[i0]
                    changed = True
    return grid

@lru_cache(None)
def opt_dist_one_line(nonogram: Tuple[int, ...], d: int) -> int:
    n = len(nonogram)
    one_count = sum(nonogram[:d])
    total_ones = sum(nonogram)
    best = (d - one_count) + (total_ones - one_count)
    current = one_count
    for i in range(1, n - d + 1):
        current += nonogram[i + d - 1] - nonogram[i - 1]
        swaps = (d - current) + (total_ones - current)
        best = min(best, swaps)
    return best

@lru_cache(None)
def opt_dist(nonogram: Tuple[int, ...], blocks: Tuple[int, ...]) -> int:
    if len(blocks) == 1:
        return opt_dist_one_line(nonogram, blocks[0])
    block = blocks[0]
    rest = blocks[1:]
    length = len(nonogram)
    max_shift = length - sum(blocks) - (len(blocks) - 1)
    best = float('inf')
    for shift in range(max_shift + 1):
        seg = nonogram[:block + shift]
        cost = opt_dist_one_line(seg, block)
        if block + shift < length and nonogram[block + shift] == FILLED:
            cost += 1
        tail_cost = opt_dist(nonogram[block + shift + 1:], rest)
        best = min(best, cost + tail_cost)
    return best


def find_wrong_line(x: int, y: int, grid: List[List[int]],
                    rows_spec: List[List[int]], cols_spec: List[List[int]]):
    wrong = set()
    for i in range(x):
        if opt_dist(tuple(grid[i]), tuple(rows_spec[i])) != 0:
            wrong.add((i, 'r'))
    for j in range(y):
        col = tuple(grid[i][j] for i in range(x))
        if opt_dist(col, tuple(cols_spec[j])) != 0:
            wrong.add((j, 'c'))
    return wrong


def find_nonogram(x: int, y: int,
                  rows_spec: List[List[int]], cols_spec: List[List[int]],
                  random_=0.1, max_time=0.1) -> List[List[int]]:
    fixed = initial_inference(x, y, rows_spec, cols_spec)
    grid = [[fixed[i][j] if fixed[i][j] != UNKNOWN else random.choice([EMPTY, FILLED])
             for j in range(y)] for i in range(x)]
    start = time.time()
    while True:
        if time.time() - start > max_time:
            grid = [[fixed[i][j] if fixed[i][j] != UNKNOWN else random.choice([EMPTY, FILLED])
                     for j in range(y)] for i in range(x)]
            start = time.time()
        wrong = find_wrong_line(x, y, grid, rows_spec, cols_spec)
        if not wrong:
            return grid
        i, t = random.choice(list(wrong))
        if t == 'r':
            line = grid[i][:]
            blocks = rows_spec[i]
        else:
            line = [grid[k][i] for k in range(x)]
            blocks = cols_spec[i]
        best_val = float('inf')
        best_idx = None
        candidates: List[int] = []
        length = len(line)
        for idx in range(length):
            if t == 'r' and fixed[i][idx] != UNKNOWN:
                continue
            if t == 'c' and fixed[idx][i] != UNKNOWN:
                continue
            new_line = line.copy()
            new_line[idx] = 1 - new_line[idx]
            if t == 'r':
                vert = [grid[r][idx] for r in range(x)]
                vert[i] = 1 - vert[i]
                vert_blocks = cols_spec[idx]
            else:
                vert = grid[idx][:]
                vert[i] = 1 - vert[i]
                vert_blocks = rows_spec[idx]
            cost = opt_dist(tuple(new_line), tuple(blocks)) + \
                   opt_dist(tuple(vert), tuple(vert_blocks))
            candidates.append(idx)
            if cost < best_val:
                best_val = cost
                best_idx = idx
        if best_idx is not None:
            if random.random() < random_:
                best_idx = random.choice(candidates)
            if t == 'r':
                grid[i][best_idx] = 1 - grid[i][best_idx]
            else:
                grid[best_idx][i] = 1 - grid[best_idx][i]


if __name__ == '__main__':
    with open('zad_input.txt', 'r') as f:
        lines = f.readlines()
    x, y = map(int, lines[0].split())
    rows_spec = [list(map(int, lines[i+1].split())) for i in range(x)]
    cols_spec = [list(map(int, lines[x+1+j].split())) for j in range(y)]

    board = find_nonogram(x, y, rows_spec, cols_spec, random_=0.1, max_time=0.1)

    with open('zad_output.txt', 'w') as f:
        for row in board:
            line = ''.join('#' if cell == FILLED else '.' for cell in row)
            f.write(line + '\n')
