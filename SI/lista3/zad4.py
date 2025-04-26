def B(i,j):
    return f'B_{i}_{j}'


def writeln(output, s):
    output.write(s + '\n')


def storms(rows, cols, triples, output):
    writeln(output, ':- use_module(library(clpfd)).')
    
    R = len(rows)
    C = len(cols)
    
    vars = [B(i,j) for i in range(R) for j in range(C)]
    
    writeln(output, 'solve([' + ', '.join(vars) + ']) :- ')
    
    writeln(output, '[' +  ', '.join(vars) + '] ins 0..1,')

    # constraints for cols and rows to match sum
    for i, row_spec in enumerate(rows):
        row_vars = ', '.join(B(i, j) for j in range(C))
        writeln(output, f'sum([{row_vars}], #=, {row_spec}),')
    
    for i, col_spec in enumerate(cols):
        col_vars = ', '.join(B(j, i) for j in range(R))
        writeln(output, f'sum([{col_vars}], #=, {col_spec}),')

    # based on input domains
    for i, j, v in triples:
        writeln(output, f'{B(i,j)} #= {v},')
    
    # at least one vertical and horizontal neighbor
    for i in range(R):
        for j in range(C):
            v = B(i, j)
            neigh_h = []
            if j > 0:
                neigh_h.append(B(i, j - 1))
            if j < C - 1:
                neigh_h.append(B(i, j + 1))
            
            if neigh_h:
                writeln(output, f'{v} #=< ' + '+'.join(neigh_h) + ',')
            
            neigh_v = []
            if i > 0:
                neigh_v.append(B(i - 1, j))
            if i < R - 1:
                neigh_v.append(B(i + 1, j))
            
            if neigh_v:
                writeln(output, f'{v} #=< ' + '+'.join(neigh_v) + ',')
            
    # no corner touch and L shape
    for i in range(R - 1):
        for j in range(C - 1):
            a = B(i, j)
            b = B(i + 1, j)
            c = B(i, j + 1)
            d = B(i + 1, j + 1)

            writeln(output, f'{a} + {d} + 1 #>= {c} + {b},')
            writeln(output, f'{b} + {c} + 1 #>= {a} + {d},')
            writeln(output, f'{a} + {b} + {c} + {d} #\\= 3,')


    writeln(output, '    labeling([ff], [' +  ', '.join(vars) + ']).' )
    writeln(output, '')
    writeln(output, ":- tell('prolog_result.txt'), solve(X), write(X), nl, told.")           


def main():
    with open('zad_input.txt') as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    
    rows = list(map(int, lines[0].split()))
    cols = list(map(int, lines[1].split()))
    triples = [tuple(map(int, ln.split())) for ln in lines[2:]]

    with open('zad_output.txt', 'w') as output:
        storms(rows, cols, triples, output)


if __name__ == '__main__':
    main()
