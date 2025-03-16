import math
from typing import List


def blotkarz_hands():
    # royal flush - no chance B and F
    # straight flush
    s_flush = 5 * 4
    # quads
    quads = 32 * 9
    # full
    full = math.comb(4, 3) * math.comb(4, 2) * 9 * 8
    # flush
    flush = math.comb(9, 5) * 4 - 20
    # straight
    straight = 5 * 4**5 - 20
    # three
    three = 9 * math.comb(4, 3) * math.comb(8, 2) * 4**2
    # two_pairs
    two_pairs = math.comb(9, 2) * math.comb(4, 2) * math.comb(4, 2) * 7 * 4
    # pair - actually useless because blotkarz can't win when he has pair
    pair = 9 * math.comb(4, 2) * math.comb(8, 3) * 4**3

    return [s_flush, quads, full, flush, straight, three, two_pairs, pair]


def figurant_hands():
    # straight flush
    s_flush = 0
    # quads
    quads = 12 * 4
    # full
    full = math.comb(4, 3) * math.comb(4, 2) * 4 * 3
    # flush
    flush = 0
    # straight
    straight = 0
    # three
    three = 4 * math.comb(4, 3) * math.comb(3, 2) * 4**2
    # two_pairs
    two_pairs = math.comb(4, 2)**3 * math.comb(2, 1) * 4
    # pair
    pair = math.comb(4, 1) * math.comb(4, 2) * 4**3

    return [s_flush, quads, full, flush, straight, three, two_pairs, pair]


def count_blotkarz_win_chance(blotkarz: List[int], figurant: List[int], all: int) -> int:
    count_blotkarz_wins = 0
    for i in range(7):
        if blotkarz[i] == 0:
            continue
        for j in range(i + 1, 8):
            count_blotkarz_wins += blotkarz[i] * figurant[j]

    return round(count_blotkarz_wins / all_possibilities * 100, 3)


if __name__ == '__main__':
    number_blotkarz_hands = math.comb(36, 5)
    number_figurant_hands = math.comb(16, 5)
    all_possibilities = number_figurant_hands * number_blotkarz_hands

    print(f'Blotkarz all: {number_blotkarz_hands}, figurant all: {number_figurant_hands}')
    print(f'All possibilities: {all_possibilities}')

    print(count_blotkarz_win_chance(blotkarz_hands(), figurant_hands(), all_possibilities))
