from collections import deque
import chess
import argparse


def map_position(pos: str) -> int:
    col = ord(pos[0]) - ord('a')
    row = int(pos[1]) - 1
    return row * 8 + col


def map_number(num: int) -> str:
    row = num // 8
    col = num % 8
    return chr(col + ord('a')) + str(row + 1)


def new_position(pos: int) -> tuple[int, int]:
    row = pos // 8
    col = pos % 8
    return row, col


def are_figures_adjacent(wk: int, bk: int) -> bool:
    wk_row = wk // 8
    wk_col = wk % 8
    bk_row = bk // 8
    bk_col = bk % 8
    if (abs(wk_row - bk_row) <= 1 and abs(wk_col - bk_col) <= 1):
        return True
    return False


def is_tower_in_danger(wk: int, wt: int, bk: int) -> bool:
    if are_figures_adjacent(wt, bk) and not are_figures_adjacent(wk, wt):
        return True
    return False


def is_check(wk: int, wt: int, bk: int) -> bool:
    wk_row = wk // 8
    wk_col = wk % 8
    wt_row = wt // 8
    wt_col = wt % 8
    bk_row = bk // 8
    bk_col = bk % 8
    
    if wt_row == bk_row:
        if wk_row == wt_row and min(wt_col, bk_col) < wk_col < max(wt_col, bk_col):
            return False
        return True
    elif wt_col == bk_col:
        if wk_col == wt_col and min(wt_row, bk_row) < wk_row < max(wt_row, bk_row):
            return False
        return True

    return False


def is_mate(wk: int, wt: int, bk: int) -> bool:
    if not is_check(wk, wt, bk):
        return False
    if is_tower_in_danger(wk, wt, bk):
        return False
    
    bk_row = bk // 8
    bk_col = bk % 8
    
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            
            new_bk_row = bk_row + i
            new_bk_col = bk_col + j
            if not (0 <= new_bk_row < 8 and 0 <= new_bk_col < 8):
                continue
            
            new_bk = new_bk_row * 8 + new_bk_col
            if are_figures_adjacent(wk, new_bk):
                continue

            if not is_check(wk, wt, new_bk):
                return False

    return True


def same_field(wk: int, wt: int, bk: int):
    return (wk == wt or wk == bk or wt == bk)


def reconstruct_path(initial_state, dist, child):
    if dist[initial_state] == -1 or dist[initial_state] == 'INF':
        return []

    path = []
    current = initial_state
    while dist[current] != 0:
        path.append(current)
        current = child[current]
    path.append(current)
    return path


def min_mate_moves(first_move: str, wk_pos: str, wt_pos: str, bk_pos: str):
    dist = {}
    prev_states = {}
    queue = deque()

    if first_move == 'black':
        is_black = 1
    else:
        is_black = 0

    wk_s = map_position(wk_pos)
    wt_s = map_position(wt_pos)
    bk_s = map_position(bk_pos)
    initial_state = (is_black, wk_s, wt_s, bk_s)

    for side in [0, 1]:
        for wk in range(64):
            for wt in range(64):
                for bk in range(64):
                    if same_field(wk, wt, bk) or are_figures_adjacent(wk, bk):
                        continue
                    
                    dist[(side, wk, wt, bk)] = -1
                    prev_states[(side, wk, wt, bk)] = []

                    if side == 1 and is_mate(wk, wt, bk):
                        dist[(1, wk, wt, bk)] = 0
                        queue.append((1, wk, wt, bk))

    # lets add to prev_states all 1-distant moves
    for side in [0, 1]:
        for wk in range(64):
            for wt in range(64):
                for bk in range(64):
                    if same_field(wk, wt, bk) or are_figures_adjacent(wk, bk):
                        continue
                    
                    # analyze 1 step moves
                    if side == 0:
                        wk_row, wk_col = new_position(wk)
                        wt_row, wt_col = new_position(wt)

                        # white king moves
                        for i in [-1, 0, 1]:
                            for j in [-1, 0, 1]:
                                if i == 0 and j == 0:
                                    continue

                                new_wk_row = wk_row + i
                                new_wk_col = wk_col + j

                                if not (0 <= new_wk_row < 8 and 0 <= new_wk_col < 8):
                                    continue

                                new_wk = new_wk_row * 8 + new_wk_col

                                if are_figures_adjacent(new_wk, bk) or same_field(new_wk, wt, bk):
                                    continue
                        
                                prev_states[(1, new_wk, wt, bk)].append((0, wk, wt, bk))
                        
                        # white tower moves
                        for i in range(8):
                            if i != wt_row:
                                new_wt = i * 8 + wt_col
                                if same_field(wk, new_wt, bk):
                                    continue
                                prev_states[(1, wk, new_wt, bk)].append((0, wk, wt, bk))

                            if i != wt_col:
                                new_wt = wt_row * 8 + i
                                if same_field(wk, new_wt, bk):
                                    continue
                                prev_states[(1, wk, new_wt, bk)].append((0, wk, wt, bk))

                    else:
                        # black king moves
                        bk_row, bk_col = new_position(bk)
                        for i in [-1, 0, 1]:
                            for j in [-1, 0, 1]:
                                if i == 0 and j == 0:
                                    continue

                                new_bk_row = bk_row + i
                                new_bk_col = bk_col + j

                                if not (0 <= new_bk_row < 8 and 0 <= new_bk_col < 8):
                                    continue

                                new_bk = new_bk_row * 8 + new_bk_col

                                if are_figures_adjacent(wk, new_bk) or same_field(wk, wt, new_bk) or is_check(wk, wt, new_bk):
                                    continue
                        
                                prev_states[(0, wk, wt, new_bk)].append((1, wk, wt, bk))

    child = {}
    while queue:
        (side, wk, wt, bk) = queue.popleft()
        d = dist[(side, wk, wt, bk)]

        for state in prev_states[(side, wk, wt, bk)]:
            if dist[state] == -1:
                dist[state] = d + 1
                queue.append(state)
                child[state] = (side, wk, wt, bk)
    
    return (dist[initial_state], initial_state, dist, child) if dist[initial_state] != -1 else 'INF'


def show_solution_on_chess_board(path, mode):
    if not path:
        if mode == 'batch':
            with open("zad1_output.txt", "w") as file:
                file.write("INF")
        else:
            print("INF")
        return

    board = chess.Board(None)
    board.clear()

    first_side, first_wk, first_wt, first_bk = path[0]

    board.set_piece_at(
        chess.parse_square(map_number(first_wk)),
        chess.Piece(chess.KING, chess.WHITE)
    )
    board.set_piece_at(
        chess.parse_square(map_number(first_wt)),
        chess.Piece(chess.ROOK, chess.WHITE)
    )
    board.set_piece_at(
        chess.parse_square(map_number(first_bk)),
        chess.Piece(chess.KING, chess.BLACK)
    )

    board.turn = chess.WHITE if first_side == 0 else chess.BLACK

    if mode == 'batch':
        file_out = open("zad1_output.txt", "w")
        file_out.write("Initial position: \n" + str(board) + "\n\n")
    else:
        print("Initial position: \n", board, "\n")

    for i in range(len(path) - 1):
        side_now, wk_now, wt_now, bk_now = path[i]
        side_next, wk_next, wt_next, bk_next = path[i+1]

        move_uci = None

        if side_now == 0:
            if wk_now != wk_next:
                move_uci = map_number(wk_now) + map_number(wk_next)
            elif wt_now != wt_next:
                move_uci = map_number(wt_now) + map_number(wt_next)
            else:
                raise ValueError("Error")
        else:
            if bk_now != bk_next:
                move_uci = map_number(bk_now) + map_number(bk_next)
            else:
                raise ValueError("Error")

        move = chess.Move.from_uci(move_uci)
        board.push(move)
        
        desc = f"Move {i+1}: {move_uci} ({'White' if side_now == 0 else 'Black'})"
            
        if mode == 'batch':
            file_out.write(desc)
        else:
            print(desc)
            print(board)
            input("Press enter to continue...")

        board.turn = chess.WHITE if side_next == 0 else chess.BLACK

    finish = "check-mate\n" + str(board) + "\n"
    if mode == 'batch':
        file_out.write(finish)
        file_out.close()
    else:
        print(finish)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=['debug', 'batch'], default='batch'
    )
    args = parser.parse_args()

    with open('zad1_input.txt', encoding='utf-8') as file:
        line = file.readline()
        first_move, white_king, white_tower, black_king = line.split()

    result, initial_state, dist, child = min_mate_moves(first_move, white_king, white_tower, black_king)

    if args.mode == 'batch':
        with open("zad1_output.txt", "w") as file:
            file.write(str(result))
            file.close()
        # path = reconstruct_path(initial_state, dist, child)
        # show_solution_on_chess_board(path, 'batch')
    elif args.mode == 'debug':
        path = reconstruct_path(initial_state, dist, child)
        show_solution_on_chess_board(path, 'debug')

    