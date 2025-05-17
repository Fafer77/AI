from typing import Dict, Tuple, List
import time
import random

ZOBRIST = {}

def get_empty_squares(one: int, two: int) -> int:
    # we do & to limit it to 64 bits
    return ~(one | two) & ((1 << 64) - 1)


def get_legal_moves(player: int, adversary: int) -> Dict[int, int]:
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    legal_moves = {}
    empty_squares = get_empty_squares(player, adversary)
    while empty_squares:
        first_empty_sq = empty_squares & -empty_squares
        sq_num = first_empty_sq.bit_length() - 1
        empty_squares &= empty_squares - 1

        flip_mask = 0
        row = sq_num // 8
        col = sq_num % 8

        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            flip_path = 0
            # we iterate as long as there is opposite color pawn
            while (0 <= new_row < 8 and 0 <= new_col < 8 and
                   (1 << (new_row * 8 + new_col)) & adversary):
                flip_path |= (1 << (new_row * 8 + new_col))
                new_row += dx
                new_col += dy
            # now we check whether there is our color pawn there to
            # jam (zakleszczyć) enemy color
            if (0 <= new_row < 8 and 0 <= new_col < 8 and
                (1 << (new_row * 8 + new_col)) & player):
                flip_mask |= flip_path
        
        if flip_mask:
            legal_moves[sq_num] = flip_mask
    
    return legal_moves


def create_new_state(player: int, adversary: int, move: int,
                     flip_mask: int) -> Tuple[int, int]:
    player |= (1 << move)
    player |= flip_mask
    adversary &= ~flip_mask

    return player, adversary


def is_terminal_state(white_board: int, black_board: int) -> bool:
    return get_empty_squares(white_board, black_board) == 0


def piece_diff(white_board: int, black_board: int) -> int:
    white_count = white_board.bit_count()
    black_count = black_board.bit_count()

    return white_count - black_count


def legal_moves_diff(white_board: int, black_board: int) -> int:
    white_legal = len(get_legal_moves(white_board, black_board))
    black_legal = len(get_legal_moves(black_board, white_board))
    
    return white_legal - black_legal


def safe_pieces(board: int) -> int:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    safe_counter = 0
    for corner in [0, 7, 56, 63]:
        if (1 << corner) & board:
            safe_counter += 1
            row = corner // 8
            col = corner % 8
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while True:
                    if (0 <= r < 8 and 0 <= c < 8 and 
                        (1 << (r * 8 + c)) & board):
                        r += dr
                        c += dc
                        safe_counter += 1
                    else:
                        break

    return safe_counter


def safe_pieces_diff(white_board: int, black_board: int) -> int:
    white_counter = safe_pieces(white_board)
    black_counter = safe_pieces(black_board)

    return white_counter - black_counter


def phase_weights(empty_count: int) -> List[int]:
    if empty_count > 48:
        w = [0.25, 0.50, 0.25]
    elif empty_count > 31:
        w = [0.25, 0.40, 0.35]
    elif empty_count > 15:
        w = [0.35, 0.25, 0.40]
    else:
        w = [0.50, 0.20, 0.30]
    return w


def evaluate(white_board: int, black_board: int) -> int:
    empty_left = get_empty_squares(white_board, black_board).bit_count()
    weights = phase_weights(empty_left)

    pieces = piece_diff(white_board, black_board)
    legals = legal_moves_diff(white_board, black_board)
    safes = safe_pieces_diff(white_board, black_board)

    return sum(w * c for w, c in zip(weights, [pieces, legals, safes]))


def init_zobrist() -> None:
    for color in ('w', 'b'):
        for sq in range(64):
            ZOBRIST[(color, sq)] = random.getrandbits(64)
    ZOBRIST['white_to_move'] = random.getrandbits(64)


def compute_zobrist(white_board: int, black_board: int, white_to_move: bool) -> int:
    h = 0
    wb = white_board
    while wb:
        sq = (wb & -wb).bit_length() - 1
        h ^= ZOBRIST[('w', sq)]
        wb &= wb - 1
    
    bb = black_board
    while bb:
        sq = (bb & -bb).bit_length() - 1
        h ^= ZOBRIST[('b', sq)]
        bb &= bb - 1
    
    if white_to_move:
        h ^= ZOBRIST['white_to_move']
    return h

# flags -> EXACT 0 , LOWER 1 (we went thorugh Beta, so we
# return lower bound), UPPER 2 -> real value is lower than this
def insert_into_tt(zkey, value, depth, flag, best_move, tt):
    old = tt.get(zkey)
    if old is None or depth >= old[1]:
        tt[zkey] = (value, depth, flag, best_move)


def lookup_tt(zkey, depth, alpha, beta, tt):
    entry = tt.get(zkey)
    if entry is None:
        return None

    value, entry_depth, flag, move = entry
    if entry_depth < depth:
        return None
    elif flag == 0:
        return value, move
    elif flag == 1 and value <= alpha:
        return alpha, move
    elif flag == 2 and value >= beta:
        return beta, move
    return None


def iterative_deepening(white_board, black_board, time_limit,
                        bot_is_white):
    white_to_move = bot_is_white
    player = white_board if bot_is_white else black_board
    adversary = black_board if bot_is_white else white_board

    best_move = None
    transposition_table = {}
    start_time = time.time()
    depth = 1

    while True:
        if time.time() - start_time >= time_limit:
            break
        try:
            val, move = minimax_alphabeta(player, adversary, depth,
                                           -float('inf'), float('inf'), start_time,
                                           time_limit, transposition_table, white_to_move,
                                           maximizing_player=True, bot_is_white=bot_is_white)
            if move is not None:
                best_move = move
        except TimeoutError:
            break
        depth += 1
    
    return best_move


def minimax_alphabeta(player, adversary, depth, alpha, beta, start, time_limit,
                      tt, white_to_move, maximizing_player, bot_is_white, passed=False):
    if depth == 0 or is_terminal_state(player, adversary):
        white_board = player if white_to_move else adversary
        black_board = adversary if white_to_move else player
        score = evaluate(white_board, black_board)

        if not bot_is_white:
            score = -score
        return score, None

    if time.time() - start >= time_limit:
        raise TimeoutError

    zkey = compute_zobrist(
        white_board = player if white_to_move else adversary,
        black_board = adversary if white_to_move else player,
        white_to_move = white_to_move
    )
    cached = lookup_tt(zkey, depth, alpha, beta, tt)
    if cached is not None:
        return cached

    moves = get_legal_moves(player, adversary)
    if not moves:
        if passed:
            return minimax_alphabeta(
                player, adversary, 0,
                alpha, beta,
                start, time_limit, tt,
                white_to_move, maximizing_player, bot_is_white, True
            ) 
        
        val, _ = minimax_alphabeta(adversary, player, depth, alpha, beta, 
                                   start,time_limit, tt, not white_to_move, 
                                   not maximizing_player, bot_is_white, True)
        return val, None

    best_move = None
    orig_alpha, orig_beta = alpha, beta

    if maximizing_player:
        best_val = -float('inf')
        for move, mask in moves.items():
            new_player, new_adv = create_new_state(player, adversary, move, mask)

            val, _ = minimax_alphabeta(
                new_adv, new_player, depth - 1,
                alpha, beta,
                start, time_limit, tt,
                not white_to_move,
                False,
                bot_is_white
            )

            if val > best_val:
                best_val, best_move = val, move
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
    else:
        best_val = float('inf')
        for move, mask in moves.items():
            new_player, new_adv = create_new_state(player, adversary, move, mask)

            val, _ = minimax_alphabeta(
                new_adv, new_player, depth - 1,
                alpha, beta,
                start, time_limit, tt,
                not white_to_move,
                True,
                bot_is_white
            )

            if val < best_val:
                best_val, best_move = val, move
            beta = min(beta, best_val)
            if beta <= alpha:
                break

    if best_val <= orig_alpha:
        flag = 2          # UPPER
    elif best_val >= orig_beta:
        flag = 1          # LOWER
    else:
        flag = 0          # EXACT
    insert_into_tt(zkey, best_val, depth, flag, best_move, tt)

    return best_val, best_move


if __name__ == '__main__':
    # bits masks to keep where white and black boards are
    white_board = (1 << 27) | (1 << 36)
    black_board = (1 << 28) | (1 << 35)

    init_zobrist()

