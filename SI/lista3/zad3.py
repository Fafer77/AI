def V(i,j):
    return 'V%d_%d' % (i,j)


def domains(Vs):
    return [ q + ' in 1..9' for q in Vs ]


def all_different(Qs):
    return 'all_distinct([' + ', '.join(Qs) + '])'


def get_column(j):
    return [V(i,j) for i in range(9)] 


def get_row(i):
    return [V(i,j) for j in range(9)] 


def horizontal():
    return [ all_different(get_row(i)) for i in range(9)]


def vertical():
    return [all_different(get_column(j)) for j in range(9)]


def block(bi, bj):
    return [V(3*bi + di, 3*bj + dj)
            for di in range(3) for dj in range(3)]

def print_constraints(Cs, output, indent, width):
    position = indent
    output.write(' ' * indent)
    for c in Cs:
        output.write(c + ', ')
        position += len(c)
        if position > width:
            output.write('\n' + ' ' * indent)
            position = indent
    output.write('\n')


def writeln(output, s):
    output.write(s + '\n')


def sudoku(assignments, output):
    variables = [ V(i,j) for i in range(9) for j in range(9)]
    
    writeln(output, ':- use_module(library(clpfd)).')
    writeln(output, 'solve([' + ', '.join(variables) + ']) :- ')
    
    Cs = []
    Cs += domains(variables)

    for i in range(9):
        Cs.append(all_different(get_row(i)))
    for j in range(9):
        Cs.append(all_different(get_column(j)))
    for bi in range(3):
        for bj in range(3):
            Cs.append(all_different(block(bi, bj)))
    
    for i, j, v in assignments:
        Cs.append(f'{V(i, j)} #= {v}')

    print_constraints(Cs, output, 4, 70),
    writeln(output, '    labeling([ff], [' +  ', '.join(variables) + ']).' )
    writeln(output, '')
    writeln(output, ":- tell('prolog_result.txt'), solve(X), write(X), nl, told.")       


if __name__ == "__main__":
    lines = []
    with open('zad_input.txt') as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) == 9:
                lines.append(line)
    
    domain = []
    for i, line in enumerate(lines[:9]):
        for j, ch in enumerate(line):
            if ch != '.':
                domain.append((i, j, int(ch)))
    
    with open('zad_output.txt', 'w') as output:
        sudoku(domain, output)
    
"""
89.356.1.
3...1.49.
....2985.
9.7.6432.
.........
.6389.1.4
.3298....
.78.4....
.5.637.48

53..7....
6..195...
.98....6.
8...6...3
4..8.3..1
7...2...6
.6....28.
...419..5
....8..79

3.......1
4..386...
.....1.4.
6.924..3.
..3......
......719
........6
2.7...3..
"""    
