from typing import List
import random
from functools import lru_cache
import time


def opt_dist_one_line(nonogram: List[int], d: int) -> int:
    n = len(nonogram)
    one_count_window = sum(nonogram[i] for i in range(d))
    total_one_count = sum(nonogram[i] for i in range(n))
    swap_count = d - one_count_window
    min_swap_count = swap_count + (total_one_count - one_count_window)

    for i in range(1, n - d + 1):
        if nonogram[i - 1] == 1:
            one_count_window -= 1
        if nonogram[i + d - 1] == 1:
            one_count_window += 1
        swap_count = d - one_count_window + (total_one_count - one_count_window)
        min_swap_count = min(min_swap_count, swap_count)

    return min_swap_count


@lru_cache(None)
def opt_dist(nonogram, blocks) -> int:
    if len(blocks) == 1:
        return opt_dist_one_line(list(nonogram), blocks[0])
    
    block = blocks[0]
    end = len(nonogram) - sum(blocks) - (len(blocks) - 1)
    lowest_cost = float('inf')
    best_pos = None

    for i in range(end + 1):
        segment = list(nonogram[:block + i])
        cost = opt_dist_one_line(segment, block)
        if block + i < len(nonogram) and nonogram[block + i] == 1:
            cost += 1
        if cost < lowest_cost:
            lowest_cost = cost
            best_pos = block + i + 1
    
    if best_pos is None:
        best_pos = len(nonogram)
    return lowest_cost + opt_dist(nonogram[best_pos:], blocks[1:])


def find_wrong_line(x: int, y: int, nono: List[List[int]], rows_spec: List[List[int]], cols_spec: List[List[int]]):
    wrong = set()
    for i in range(x):
        if opt_dist(tuple(nono[i]), tuple(rows_spec[i])) != 0:
            wrong.add((i, 'r'))
    for j in range(y):
        col = [nono[i][j] for i in range(x)]
        if opt_dist(tuple(col), tuple(cols_spec[j])) != 0:
            wrong.add((j, 'c'))
    return wrong


def find_nonogram(x: int, y: int, rows_spec: List[List[int]], cols_spec: List[List[int]],
                  random_ = 0.1, max_time = 0.1) -> List[List[int]]:
    while True:
        nonogram = [[random.choice([0, 1]) for _ in range(y)] for _ in range(x)]
        start = time.time()
        while time.time() - start < max_time:
            wrong = find_wrong_line(x, y, nonogram, rows_spec, cols_spec)
            if not wrong:
                return nonogram
            idx, t = random.choice(list(wrong))
            if t == 'r':
                line = nonogram[idx][:]
                blocks = rows_spec[idx]
            else:
                line = [nonogram[k][idx] for k in range(x)]
                blocks = cols_spec[idx]
            choice = []
            best_val = float('inf')
            best_candidate = -1
            for i in range(len(line)):
                if t == 'r':
                    vertical_line = [nonogram[j][i] for j in range(x)]
                    vertical_blocks = cols_spec[i]
                else:
                    vertical_line = nonogram[i][:]
                    vertical_blocks = rows_spec[i]
                line[i] = 1 - line[i]
                vertical_line[idx] = 1 - vertical_line[idx]
                changed_val = opt_dist(tuple(line), tuple(blocks))
                changed_vert_val = opt_dist(tuple(vertical_line), tuple(vertical_blocks))
                total_val = changed_val + changed_vert_val
                choice.append(i)
                if total_val < best_val:
                    best_val = total_val
                    best_candidate = i
                line[i] = 1 - line[i]
                vertical_line[idx] = 1 - vertical_line[idx]
            if best_candidate != -1:
                if random.random() < random_:
                    best_candidate = random.choice(choice)
                if t == 'r':
                    nonogram[idx][best_candidate] = 1 - nonogram[idx][best_candidate]
                else:
                    nonogram[best_candidate][idx] = 1 - nonogram[best_candidate][idx]


if __name__ == '__main__':
    with open('zad_input.txt', 'r') as file:
        lines = file.readlines()
    x, y = map(int, lines[0].split())
    rows_spec = [list(map(int, line.strip().split())) for line in lines[1:x+1]]
    cols_spec = [list(map(int, line.strip().split())) for line in lines[x+1:x+1+y]]
    
    board = find_nonogram(x, y, rows_spec, cols_spec, random_=0.1, max_time=0.1)
    
    with open('zad_output.txt', 'w') as f:
        for i in range(x):
            line = "".join("#" if board[i][j] == 1 else "." for j in range(y))
            f.write(line + "\n")