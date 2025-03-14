# zgodnie z algorytmem z polecenia zadania

from typing import List
import random


def opt_dist(nonogram: List[int], d: int) -> int:
    if d == 0:
        return sum(nonogram)

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


def change_bit(bit: int) -> int:
    return 1 if bit == 0 else 0


def get_flip_improvement(board: List[List[int]], row: int, col: int,
                         rows_spec: List[int], cols_spec: List[int]) -> int:

    old_row_cost = opt_dist(board[row], rows_spec[row])
    old_col_bits = [board[r][col] for r in range(len(board))]
    old_col_cost = opt_dist(old_col_bits, cols_spec[col])

    old_bit = board[row][col]
    board[row][col] = change_bit(old_bit)

    new_row_cost = opt_dist(board[row], rows_spec[row])
    new_col_bits = [board[r][col] for r in range(len(board))]
    new_col_cost = opt_dist(new_col_bits, cols_spec[col])

    board[row][col] = old_bit

    return (old_row_cost + old_col_cost) - (new_row_cost + new_col_cost)


def update_cost(board: List[List[int]], to_fix_rows: set[int], to_fix_cols: set[int], 
                x: int, y: int, rows_spec: List[int], cols_spec: List[int]) -> int:
    total_cost = 0
    for i in range(x):
        cost = opt_dist(board[i], rows_spec[i])
        if cost != 0:
            total_cost += cost
            to_fix_rows.add(i)
        else:
            if i in to_fix_rows:
                to_fix_rows.remove(i)

    for i in range(y):
        col = [board[j][i] for j in range(x)]
        cost = opt_dist(col, cols_spec[i])
        if cost != 0:
            total_cost += cost
            to_fix_cols.add(i)
        else:
            if i in to_fix_cols:
                to_fix_cols.remove(i)

    return total_cost


def find_simple_nonogram(rows_spec: List[int], cols_spec: List[int], x: int, y: int) -> List[List[int]]:
    while True:
        board = [[random.randint(0, 1) for _ in range(y)] for _ in range(x)]

        total_cost = 0
        to_fix_rows = set()
        to_fix_cols = set()

        for i in range(x):
            cost = opt_dist(board[i], rows_spec[i])
            if cost != 0:
                total_cost += cost
                to_fix_rows.add(i)

        for i in range(y):
            col = [board[j][i] for j in range(x)]
            cost = opt_dist(col, cols_spec[i])
            if cost != 0:
                total_cost += cost
                to_fix_cols.add(i)

        for i in range(100_000):
            if total_cost == 0:
                return board

            random_number = random.random()
            if random_number <= 0.05 or (not to_fix_cols and not to_fix_rows):
                row = random.randint(0, x - 1)
                col = random.randint(0, y - 1)
                bit = board[row][col]
                board[row][col] = change_bit(bit)
            else:
                ax = random.randint(0, 1)
                if (ax == 0 and to_fix_rows) or (not to_fix_cols):
                    row = random.choice(list(to_fix_rows))
                    best_col = None
                    best_delta = float("-inf")
                    for c in range(y):
                        delta = get_flip_improvement(board, row, c, rows_spec, cols_spec)
                        if delta > best_delta:
                            best_delta = delta
                            best_col = c

                    bit = board[row][best_col]
                    board[row][best_col] = change_bit(bit)
                else:
                    col = random.choice(list(to_fix_cols))
                    best_row = None
                    best_delta = float("-inf")
                    for r in range(x):
                        delta = get_flip_improvement(board, r, col, rows_spec, cols_spec)
                        if delta > best_delta:
                            best_delta = delta
                            best_row = r
                    bit = board[best_row][col]
                    board[best_row][col] = change_bit(bit)
            
            # update the cost
            total_cost = update_cost(board, to_fix_rows, to_fix_cols, x, y,
                                     rows_spec, cols_spec)


if __name__ == '__main__':
    rows_spec = []
    cols_spec = []
    with open('zad5_input.txt', 'r') as file:
        x, y = map(int, file.readline().split())
        
        for _ in range(x):
            row_value = int(file.readline().strip())
            rows_spec.append(row_value)

        for _ in range(y):
            col_value = int(file.readline().strip())
            cols_spec.append(col_value)

    board = find_simple_nonogram(rows_spec, cols_spec, x, y)


    with open('zad5_output.txt', 'w') as f:
        for i in range(x):
            line = "".join("#" if board[i][j] == 1 else "." for j in range(y))
            f.write(line + "\n")

