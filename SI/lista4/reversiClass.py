import time
import random


class ReversiPSK:
    """Prosty bot Reversi (minimax + αβ + iteracyjne pogłębianie)."""

    _ZOBRIST = {}

    # ------------------------------------------------------------------ #
    #  KONSTRUKTOR
    # ------------------------------------------------------------------ #
    def __init__(self, bot_is_white, time_limit=2.0):
        self.bot_is_white = bot_is_white
        self.time_limit   = time_limit
        self._init_zobrist()

    # ------------------------------------------------------------------ #
    #  PUBLICZNE API
    # ------------------------------------------------------------------ #
    def choose_move(self, white_board, black_board):
        """Zwraca indeks pola (0-63) albo None, gdy brak ruchów bota."""
        return self._iterative_deepening(
            white_board, black_board, self.time_limit, self.bot_is_white
        )

    # ------------------------------------------------------------------ #
    #  POMOCNICZE BITBOARDY
    # ------------------------------------------------------------------ #
    def _get_empty_squares(self, one, two):
        return ~(one | two) & ((1 << 64) - 1)

    def _create_new_state(self, player, adversary, move, flip_mask):
        player   |= (1 << move) | flip_mask
        adversary &= ~flip_mask
        return player, adversary

    # ------------------------------------------------------------------ #
    #  RUCHY
    # ------------------------------------------------------------------ #
    def _get_legal_moves(self, player, adversary):
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        legal = {}
        empty = self._get_empty_squares(player, adversary)

        while empty:
            sq = (empty & -empty).bit_length() - 1
            empty &= empty - 1
            row, col = divmod(sq, 8)
            flip_mask = 0

            for dx, dy in directions:
                r, c = row + dx, col + dy
                path = 0
                while (0 <= r < 8 and 0 <= c < 8 and
                       ((1 << (r*8 + c)) & adversary)):
                    path |= (1 << (r*8 + c))
                    r += dx
                    c += dy
                if (0 <= r < 8 and 0 <= c < 8 and
                        ((1 << (r*8 + c)) & player)):
                    flip_mask |= path

            if flip_mask:
                legal[sq] = flip_mask
        return legal

    # ------------------------------------------------------------------ #
    #  EWALUACJA
    # ------------------------------------------------------------------ #
    def _piece_diff(self, w, b):
        return w.bit_count() - b.bit_count()

    def _legal_moves_diff(self, w, b):
        return len(self._get_legal_moves(w, b)) - len(self._get_legal_moves(b, w))

    def _safe_pieces(self, board):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        cnt = 0
        for corner in (0, 7, 56, 63):
            if board & (1 << corner):
                cnt += 1
                row, col = divmod(corner, 8)
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    while 0 <= r < 8 and 0 <= c < 8 and board & (1 << (r*8 + c)):
                        cnt += 1
                        r += dr
                        c += dc
        return cnt

    def _safe_pieces_diff(self, w, b):
        return self._safe_pieces(w) - self._safe_pieces(b)

    def _phase_weights(self, empty_cnt):
        if   empty_cnt > 48: return (0.25, 0.50, 0.25)
        elif empty_cnt > 31: return (0.25, 0.40, 0.35)
        elif empty_cnt > 15: return (0.35, 0.25, 0.40)
        else:                return (0.50, 0.20, 0.30)

    def _evaluate(self, w, b):
        empty_cnt = self._get_empty_squares(w, b).bit_count()
        w1, w2, w3 = self._phase_weights(empty_cnt)
        return (w1 * self._piece_diff(w, b) +
                w2 * self._legal_moves_diff(w, b) +
                w3 * self._safe_pieces_diff(w, b))

    # ------------------------------------------------------------------ #
    #  ZOBRIST HASHING
    # ------------------------------------------------------------------ #
    def _init_zobrist(self):
        if ReversiPSK._ZOBRIST:
            return
        for color in ('w', 'b'):
            for sq in range(64):
                ReversiPSK._ZOBRIST[(color, sq)] = random.getrandbits(64)
        ReversiPSK._ZOBRIST['white_to_move'] = random.getrandbits(64)

    def _compute_zobrist(self, w, b, white_to_move):
        h = 0
        wb = w
        while wb:
            sq = (wb & -wb).bit_length() - 1
            h ^= self._ZOBRIST[('w', sq)]
            wb &= wb - 1
        bb = b
        while bb:
            sq = (bb & -bb).bit_length() - 1
            h ^= self._ZOBRIST[('b', sq)]
            bb &= bb - 1
        if white_to_move:
            h ^= self._ZOBRIST['white_to_move']
        return h

    # ------------------------------------------------------------------ #
    #  TRANSPOSITION TABLE
    # ------------------------------------------------------------------ #
    def _insert_tt(self, zkey, value, depth, flag, best_move, tt):
        old = tt.get(zkey)
        if old is None or depth >= old[1]:
            tt[zkey] = (value, depth, flag, best_move)

    def _lookup_tt(self, zkey, depth, alpha, beta, tt):
        entry = tt.get(zkey)
        if entry is None or entry[1] < depth:
            return None
        value, _, flag, move = entry
        if   flag == 0: return value, move
        elif flag == 1 and value <= alpha: return alpha, move
        elif flag == 2 and value >= beta:  return beta,  move
        return None

    # ------------------------------------------------------------------ #
    #  ITERATIVE DEEPENING
    # ------------------------------------------------------------------ #
    def _iterative_deepening(self, w, b, time_limit, bot_is_white):
        player, adversary = (w, b) if bot_is_white else (b, w)
        white_to_move = bot_is_white   # bot zawsze wywołuje w swojej turze

        best_move = None
        tt = {}
        start = time.time()
        depth = 1

        while time.time() - start < time_limit:
            try:
                _, move = self._minimax_alphabeta(
                    player, adversary, depth,
                    -float('inf'), float('inf'),
                    start, time_limit,
                    tt, white_to_move,
                    maximizing=True,
                    bot_is_white=bot_is_white
                )
                if move is not None:
                    best_move = move
            except TimeoutError:
                break
            depth += 1
        return best_move

    # ------------------------------------------------------------------ #
    #  MINIMAX + αβ
    # ------------------------------------------------------------------ #
    def _minimax_alphabeta(self, player, adversary,
                           depth, alpha, beta,
                           start, time_limit,
                           tt, white_to_move,
                           maximizing, bot_is_white,
                           passed=False):

        # --- koniec gry / maks. głębokość / timeout --------------------
        if depth == 0 or self._get_empty_squares(player, adversary) == 0:
            w_board = player if white_to_move else adversary
            b_board = adversary if white_to_move else player
            score = self._evaluate(w_board, b_board)
            if not bot_is_white:
                score = -score
            return score, None

        if time.time() - start >= time_limit:
            raise TimeoutError

        # --- transposition table --------------------------------------
        zkey = self._compute_zobrist(
            player if white_to_move else adversary,
            adversary if white_to_move else player,
            white_to_move
        )
        cached = self._lookup_tt(zkey, depth, alpha, beta, tt)
        if cached is not None:
            return cached

        # --- legal moves / pass ---------------------------------------
        moves = self._get_legal_moves(player, adversary)
        if not moves:
            if passed:   # dwa pasy = koniec
                return self._minimax_alphabeta(
                    player, adversary, 0,
                    alpha, beta,
                    start, time_limit, tt,
                    white_to_move, maximizing, bot_is_white, True
                )
            # pojedynczy pass
            val, _ = self._minimax_alphabeta(
                adversary, player, depth,
                alpha, beta,
                start, time_limit, tt,
                not white_to_move, not maximizing, bot_is_white, True
            )
            return val, None

        # --- rekursja --------------------------------------------------
        best_move = None
        orig_alpha, orig_beta = alpha, beta

        if maximizing:
            best_val = -float('inf')
            for move, mask in moves.items():
                np, na = self._create_new_state(player, adversary, move, mask)
                val, _ = self._minimax_alphabeta(
                    na, np, depth-1,
                    alpha, beta,
                    start, time_limit, tt,
                    not white_to_move, False, bot_is_white
                )
                if val > best_val:
                    best_val, best_move = val, move
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
        else:
            best_val = float('inf')
            for move, mask in moves.items():
                np, na = self._create_new_state(player, adversary, move, mask)
                val, _ = self._minimax_alphabeta(
                    na, np, depth-1,
                    alpha, beta,
                    start, time_limit, tt,
                    not white_to_move, True, bot_is_white
                )
                if val < best_val:
                    best_val, best_move = val, move
                beta = min(beta, best_val)
                if beta <= alpha:
                    break

        # --- zapis do TT ----------------------------------------------
        flag = 0
        if best_val <= orig_alpha: flag = 2      # UPPER
        elif best_val >= orig_beta: flag = 1     # LOWER
        self._insert_tt(zkey, best_val, depth, flag, best_move, tt)

        return best_val, best_move


class RandomCornersBot:
    CORNERS = {0, 7, 56, 63}

    def __init__(self, bot_is_white):
        self.bot_is_white = bot_is_white          # True - gra białymi

    # ----------------------------- bit-board helpers ------------------
    def _get_empty_squares(self, one, two):
        return ~(one | two) & ((1 << 64) - 1)

    def _get_legal_moves(self, player, adversary):
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        legal = {}
        empty = self._get_empty_squares(player, adversary)

        while empty:
            sq = (empty & -empty).bit_length() - 1
            empty &= empty - 1
            row, col = divmod(sq, 8)
            flip_mask = 0

            for dx, dy in directions:
                r, c = row + dx, col + dy
                path = 0
                while (0 <= r < 8 and 0 <= c < 8 and
                       ((1 << (r*8 + c)) & adversary)):
                    path |= (1 << (r*8 + c))
                    r += dx
                    c += dy
                if (0 <= r < 8 and 0 <= c < 8 and
                        ((1 << (r*8 + c)) & player)):
                    flip_mask |= path

            if flip_mask:
                legal[sq] = flip_mask
        return legal

    # ----------------------------- public API -------------------------
    def choose_move(self, white_board, black_board):
        player   = white_board if self.bot_is_white else black_board
        adversary = black_board if self.bot_is_white else white_board

        moves = list(self._get_legal_moves(player, adversary))
        if not moves:
            return None     # brak ruchów → „pass”

        corner_moves = [m for m in moves if m in self.CORNERS]
        return random.choice(corner_moves or moves)


# ---------------------------------------------------------------------
#  2.  Funkcje do symulacji partii między dwoma botami bit-boardowymi
# ---------------------------------------------------------------------
def _create_new_state(player, adversary, move, flip_mask):
    player   |= (1 << move) | flip_mask
    adversary &= ~flip_mask
    return player, adversary


def _legal_moves_with_masks(player, adversary):
    """Zwróć słownik move→flip_mask (kopiujemy z ReversiPSK)."""
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    legal = {}
    empty = ~(player | adversary) & ((1 << 64) - 1)
    while empty:
        sq = (empty & -empty).bit_length() - 1
        empty &= empty - 1
        row, col = divmod(sq, 8)
        flip_mask = 0
        for dx, dy in directions:
            r, c = row + dx, col + dy
            path = 0
            while (0 <= r < 8 and 0 <= c < 8 and
                   ((1 << (r*8 + c)) & adversary)):
                path |= 1 << (r*8 + c)
                r += dx
                c += dy
            if (0 <= r < 8 and 0 <= c < 8 and
                    ((1 << (r*8 + c)) & player)):
                flip_mask |= path
        if flip_mask:
            legal[sq] = flip_mask
    return legal


def play_single_game(bot_white, bot_black, verbose=False):
    """Zwraca +1 jeśli wygra biały, -1 jeśli czarny, 0 = remis."""
    white = (1 << 27) | (1 << 36)
    black = (1 << 28) | (1 << 35)
    white_to_move = True
    passes = 0

    while passes < 2:
        player = white if white_to_move else black
        adversary = black if white_to_move else white
        legal = _legal_moves_with_masks(player, adversary)

        if not legal:               # brak ruchu → pass
            passes += 1
            white_to_move = not white_to_move
            continue
        passes = 0

        move = (bot_white if white_to_move else bot_black).choose_move(white, black)
        if move is None or move not in legal:
            move = random.choice(list(legal))  # awaryjna losowość

        flip_mask = legal[move]
        player, adversary = _create_new_state(player, adversary, move, flip_mask)
        if white_to_move:
            white, black = player, adversary
        else:
            black, white = player, adversary

        white_to_move = not white_to_move

    score = white.bit_count() - black.bit_count()
    if verbose:
        print("Koniec gry – wynik (białe-czarne):", score)
    return 1 if score > 0 else (-1 if score < 0 else 0)


def simulate_games(n_games, time_limit=2.0):
    """
    Rozgrywa n gier ReversiPSK ↔ RandomCornersBot.
    W każdej partii losujemy, kto gra białymi.
    """
    results = {"ReversiPSK": 0, "Random": 0, "Draw": 0}

    for _ in range(n_games):
        # losujemy kolory
        psk_is_white = random.choice([True, False])

        if psk_is_white:
            bot_white = ReversiPSK(True,  time_limit)
            bot_black = RandomCornersBot(False)
        else:
            bot_white = RandomCornersBot(True)
            bot_black = ReversiPSK(False, time_limit)

        outcome = play_single_game(bot_white, bot_black)
        print(results)
        # outcome:  1 = wygrywa biały, -1 = czarny, 0 = remis
        if outcome == 1:          # wygrał biały
            if psk_is_white:
                results["ReversiPSK"] += 1
            else:
                results["Random"]     += 1
        elif outcome == -1:       # wygrał czarny
            if psk_is_white:
                results["Random"]     += 1
            else:
                results["ReversiPSK"] += 1
        else:
            results["Draw"] += 1

    print(f"\nPo {n_games} grach:")
    print("  zwycięstwa ReversiPSK :", results['ReversiPSK'])
    print("  zwycięstwa Random     :", results['Random'])
    print("  remisy                :", results['Draw'])


# ---------------------------------------------------------------------
#  3.  Uruchom prosty test, jeśli plik wykonany bezpośrednio
# ---------------------------------------------------------------------
if __name__ == '__main__':
    simulate_games(100, time_limit=0.01)
