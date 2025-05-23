#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import random
import sys

class Jungle:
    COLOR_WHITE = 1
    COLOR_BLACK = 0
    WIDTH, HEIGHT = 7, 9
    PIECE_TYPES = ['R', 'C', 'D', 'W', 'J', 'T', 'L', 'E']

    STRENGTH = {
        'R': 1,
        'C': 2,
        'D': 3,
        'W': 4,
        'J': 5,
        'T': 6,
        'L': 7,
        'E': 8,
    }

    _MAP = [
        "..#*#..",
        "...#...",
        ".......",
        ".~~.~~.",
        ".~~.~~.",
        ".~~.~~.",
        ".......",
        "...#...",
        "..#*#..",
    ]

    def __init__(self):
        # array of bitboards - first 8 for black, another 8 for white
        self.bb = [[0] * len(self.PIECE_TYPES) for _ in range(2)]

        self.river = self._make_mask('~') # water
        self.traps = self._make_mask('#') # traps
        self.dens = self._make_mask('*') # dens
        full_mask = (1 << (self.WIDTH * self.HEIGHT)) - 1
        # meadows are just the rest of mask
        self.meadows = full_mask & ~(self.river | self.traps | self.dens)

        start_positions = {
            # dolne, małe litery – BLACK zaczyna
            self.COLOR_BLACK: {
                'L': (8, 6), 'T': (8, 0), 'J': (6, 4), 'W': (6, 2),
                'D': (7, 5), 'C': (7, 1), 'R': (6, 6), 'E': (6, 0)
            },
            # górne, wielkie litery – WHITE
            self.COLOR_WHITE: {
                'L': (0, 0), 'T': (0, 6), 'J': (2, 2), 'W': (2, 4),
                'D': (1, 1), 'C': (1, 5), 'R': (2, 0), 'E': (2, 6)
            }
        }
        for color, pos_d in start_positions.items():
            for sym, (r, c) in pos_d.items():
                t = self.PIECE_TYPES.index(sym)
                self.set_piece(color, t, r, c)


    def _idx(self, row, col):
        # returns index of bit (0..WIDTH*HEIGHT-1) for field (row, col)
        return row * self.WIDTH + col

    # create mask for particular symbol
    def _make_mask(self, symbol):
        mask = 0
        for r, line in enumerate(self._MAP):
            for c, ch in enumerate(line):
                if ch == symbol:
                    mask |= 1 << self._idx(r, c)
        return mask

    # check whether there is figure of player on this field
    def is_white_piece(self, row, col):
        idx = self._idx(row, col)
        occupied = 0
        for fig in range(len(self.PIECE_TYPES)):
            occupied |= self.bb[self.COLOR_WHITE][fig]
        
        return (occupied >> idx) & 1 == 1

    def is_black_piece(self, row, col):
        idx = self._idx(row, col)
        occupied = 0
        for fig in range(len(self.PIECE_TYPES)):
            occupied |= self.bb[self.COLOR_BLACK][fig]
        
        return (occupied >> idx) & 1 == 1

    def set_piece(self, color, fig_type, row, col):
        idx = self._idx(row, col)
        self.bb[color][fig_type] |= (1 << idx)
    
    def remove_piece(self, color, fig_type, row, col):
        idx = self._idx(row, col)
        self.bb[color][fig_type] &= ~(1 << idx)
    
    def is_terminal_state(self):
        black_den = 1 << self._idx(0, 3)
        white_den = 1 << self._idx(8, 3)
        if self.get_occupied_fields(self.COLOR_WHITE) & black_den:
            return True
        if self.get_occupied_fields(self.COLOR_BLACK) & white_den:
            return True
        return False

    def who_won(self):
        black_den = 1 << self._idx(0, 3)
        white_den = 1 << self._idx(8, 3)
        if self.get_occupied_fields(self.COLOR_WHITE) & black_den:
            return self.COLOR_WHITE
        elif self.get_occupied_fields(self.COLOR_BLACK) & white_den:
            return self.COLOR_BLACK
        return None

    def get_occupied_fields(self, color):
        if color is None:
            # all taken fields
            return (self.get_occupied_fields(self.COLOR_WHITE) |
                    self.get_occupied_fields(self.COLOR_BLACK))
        mask = 0
        for fig in range(len(self.PIECE_TYPES)):
            mask |= self.bb[color][fig]
        return mask
    
    def apply_move(self, color, type_idx, from_idx, to_idx):
        self.bb[color][type_idx] &= ~(1 << from_idx)
        # we have to check whether there is enemy on to_idx
        for t in range(len(self.PIECE_TYPES)):
            if (self.bb[1 - color][t] >> to_idx) & 1:
                self.bb[1 - color][t] &= ~(1 << to_idx)
                break
        self.bb[color][type_idx] |= (1 << to_idx)

    def can_beat(self, att_color, att_type, from_idx, to_idx):
        if ((self.traps >> from_idx) & 1):
            return False
        
        rival = 1 - att_color
        defender_type = None
        for t in range(len(self.PIECE_TYPES)):
            # find defender on this field
            if((self.bb[rival][t] >> to_idx) & 1):
                defender_type = t
                break
        # empty field
        if defender_type is None:
            return False

        att_symbol = self.PIECE_TYPES[att_type]
        def_symbol = self.PIECE_TYPES[defender_type]

        # rat from water can't beat
        if att_symbol == 'R' and ((self.river >> from_idx) & 1):
            return False

        # defender in trap (easy target)
        if ((self.traps >> to_idx) & 1):
            return True
        
        # rat beats elephant
        if att_symbol == 'R' and def_symbol == 'E':
            return True

        # elephant can't beat rat
        if att_symbol == 'E' and def_symbol == 'R':
            return False

        return self.STRENGTH[att_symbol] >= self.STRENGTH[def_symbol]
        
    def generate_moves(self, color):
        moves = []
        player_occupied = self.get_occupied_fields(color)
        enemy_occupied = self.get_occupied_fields(1 - color)
        rat_mask = (self.bb[self.COLOR_WHITE][0] |
                    self.bb[self.COLOR_BLACK][0])
        
        for idx, symbol in enumerate(self.PIECE_TYPES):
            bb = self.bb[color][idx]
            while bb:
                lsb = bb & -bb
                from_idx = lsb.bit_length() - 1
                bb &= bb - 1
                row, col = divmod(from_idx, self.WIDTH)

                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < self.HEIGHT and 0 <= nc < self.WIDTH):
                        continue
                    to_idx = self._idx(nr, nc)
                    tmask = 1 << to_idx # bitmask of where we're going

                    # can't enter my own den
                    if (color == self.COLOR_WHITE and nc == 3 and nr == 8) or \
                        (color == self.COLOR_BLACK and nc == 3 and nr == 0):
                        continue

                    # tiger and lion jump
                    if symbol in ('T', 'L') and ((self.river >> to_idx) & 1):
                        jump_r, jump_c = nr, nc
                        blocked = False
                        while (0 <= jump_r < self.HEIGHT and 0 
                               <= jump_c < self.WIDTH and 
                               (self.river >> self._idx(jump_r, jump_c)) & 1):
                            if (rat_mask >> self._idx(jump_r, jump_c)) & 1:
                                blocked = True
                                break
                            jump_r += dr
                            jump_c += dc
                        
                        if (not blocked and 0 <= jump_r < self.HEIGHT and
                            0 <= jump_c < self.WIDTH):
                            landing_idx = self._idx(jump_r, jump_c)
                            land_mask = 1 << landing_idx
                            if not (land_mask & player_occupied):
                                if land_mask & enemy_occupied:
                                    if self.can_beat(color, idx, from_idx, landing_idx):
                                        moves.append((idx, from_idx, landing_idx))
                                else:
                                    moves.append((idx, from_idx, landing_idx))
                        continue


                    # other than rat can't enter water
                    if symbol != 'R' and ((self.river >> to_idx) & 1):
                        continue

                    # colision with my figure
                    if tmask & player_occupied:
                        continue

                    # enemy on this field
                    if tmask & enemy_occupied:
                        if self.can_beat(color, idx, from_idx, to_idx):
                            moves.append((idx, from_idx, to_idx))
                    else:
                        moves.append((idx, from_idx, to_idx))
        return moves

    def clone_state(self):
        obj = Jungle.__new__(Jungle)
        obj.bb = [self.bb[0][:], self.bb[1][:]]
        obj.river, obj.traps, obj.dens, obj.meadows = (
            self.river, self.traps, self.dens, self.meadows
        )
        return obj
    
    def simulate(self, color):
        sim_state = self.clone_state()
        cur = color
        moves_counter = 1

        while True:
            if sim_state.is_terminal_state():
                winner = sim_state.who_won()
                return (1 if winner == color else 0), moves_counter

            moves = sim_state.generate_moves(cur)
            if not moves:
                return 0, moves_counter

            type_idx, from_idx, to_idx = random.choice(moves)
            sim_state.apply_move(cur, type_idx, from_idx, to_idx)
            moves_counter += 1
            cur = 1 - cur

    def flat_monte_carlo(self, color, N=20000):
        moves = self.generate_moves(color)
        m = len(moves)
        if m == 0:
            return None
        
        # arrays to keep stats for each move
        wins = [0] * m
        trials = [0] * m
        n = 0
        i = 0

        while n < N:
            type_idx, from_idx, to_idx = moves[i]
            result, depth_used = self.simulate(color)
            wins[i] += result
            trials[i] += 1
            n += depth_used
            i = (i + 1) % m
        
        rates = [wins[j] / trials[j] for j in range(m)]
        best = max(range(m), key=lambda j: rates[j])
        return moves[best]

class Player(object):
    def __init__(self):
        self.reset()

    def reset(self):
        # nowa gra, odświeżamy stan Jungle
        self.game = Jungle()
        # w Jungle małymi literami zaczyna gracz BLACK = 0
        # więc jeśli dostaniemy UGO, to będziemy BLACK
        self.my_player = None
        self.say('RDY')

    def say(self, what):
        sys.stdout.write(what + '\n')
        sys.stdout.flush()

    def hear(self):
        line = sys.stdin.readline().strip().split()
        return line[0], line[1:]

    def do_move(self, move):
        """
        Aplikuje move = ((xs, ys), (xd, yd)) lub None
        na wewnętrzny stan self.game.
        """
        if move is None:
            return
        (xs, ys), (xd, yd) = move
        from_idx = ys * self.game.WIDTH + xs
        to_idx   = yd * self.game.WIDTH + xd
        # znajdź typ bierki przeciwnika na from_idx
        for color in (self.game.COLOR_BLACK, self.game.COLOR_WHITE):
            for t in range(len(self.game.PIECE_TYPES)):
                if (self.game.bb[color][t] >> from_idx) & 1:
                    self.game.apply_move(color, t, from_idx, to_idx)
                    return

    def moves(self, color):
        """
        Zwraca listę ruchów w formacie [ ((xs,ys),(xd,yd)), … ]
        z Twojego bitboardowego generatora.
        """
        out = []
        for t, f, to in self.game.generate_moves(color):
            fr_y, fr_x = divmod(f, self.game.WIDTH)
            to_y, to_x = divmod(to, self.game.WIDTH)
            out.append(((fr_x, fr_y), (to_x, to_y)))
        return out

    def loop(self):
        while True:
            cmd, args = self.hear()

            if cmd == 'ONEMORE':
                self.reset()
                continue

            if cmd == 'BYE':
                break

            if cmd == 'UGO':
                # zaczynamy jako pierwsi ⇒ jesteśmy BLACK = 0
                self.my_player = self.game.COLOR_BLACK

            elif cmd == 'HEDID':
                # jeśli to nasz pierwszy HEDID, to gramy jako drudzy ⇒ WHITE = 1
                if self.my_player is None:
                    self.my_player = self.game.COLOR_WHITE

                # parsujemy ruch przeciwnika
                xs, ys, xd, yd = map(int, args[2:])
                if (xs, ys, xd, yd) == (-1, -1, -1, -1):
                    opp_move = None
                else:
                    opp_move = ((xs, ys), (xd, yd))
                self.do_move(opp_move)

            # po UGO lub HEDID idzie nasza tura
            # my_player na pewno jest już 0 lub 1
            mc = self.game.flat_monte_carlo(self.my_player)
            if mc is None:
                self.say('IDO -1 -1 -1 -1')
            else:
                t, f, to = mc
                fr_y, fr_x = divmod(f, self.game.WIDTH)
                to_y, to_x = divmod(to,   self.game.WIDTH)
                our_move = ((fr_x, fr_y), (to_x, to_y))
                self.do_move(our_move)
                self.say(f'IDO {fr_x} {fr_y} {to_x} {to_y}')


if __name__ == '__main__':
    player = Player()
    player.loop()
