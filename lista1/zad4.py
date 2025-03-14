# metoda tzw. slicing window -> liczę ile zmian wykonać w pierwszym oknie po czym je
# sukcesywnie przesuwam w stronę końca listy i liczę czy trzeba wykonać zmianę mniej czy więcej
# w stosunku do poprzedniego okna

from typing import List


def opt_dist(nonogram: List[int], d: int) -> str:
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

    return str(min_swap_count)


if __name__ == '__main__':
    answers = []
    with open('zad4_input.txt') as file:
        for line in file:
            line = line.strip()
            nonogram, d = line.split()
            nonogram = [int(bit) for bit in nonogram]
            ans = opt_dist(nonogram, int(d))
            answers.append(ans)

    with open('zad4_output.txt', 'w') as file:
        for ans in answers:
            file.write(ans + '\n')