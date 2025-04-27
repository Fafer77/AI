from typing import List
import sys
from collections import deque, defaultdict

def generate_lines(spec: List[int], n: int) -> List[List[str]]:
    spec_len = len(spec)
    result = []

    if not spec:
        return ['0' * n] 

    def backtrack(spec_i, start_pos, curr_line):
        if spec_i == spec_len:
            result.append("".join(curr_line))
            return
    
        block_len = spec[spec_i]
        for i in range(start_pos, n - block_len + 1):
            new_line = curr_line[:]
            for j in range(i, i + block_len):
                new_line[j] = '1'
            
            next_start = i + block_len + 1
            backtrack(spec_i + 1, next_start, new_line)

    initial_line = ['0'] * n
    backtrack(0, 0, initial_line)

    return result


def full_line(spec, i, n, domain, line_type, tuple_counter, patterns):
    if (sum(spec) + len(spec) - 1 == n):
        perfect_pattern = []
        idx = 0
        for block_len in spec:
            for j in range(idx, idx + block_len):
                if line_type == 'C':
                    if domain[j][i] == ('0', '1'):
                        domain[j][i] = '1'
                        tuple_counter -= 1
                else:
                    if domain[i][j] == ('0', '1'):
                        domain[i][j] = '1'
                        tuple_counter -= 1
                perfect_pattern.append('1')

            temp = idx + block_len
            if line_type == 'C' and temp < n and domain[temp][i] == ('0', '1'):
                domain[temp][i] = '0'
                tuple_counter -= 1
            elif line_type == 'R' and temp < n and domain[i][temp] == ('0', '1'):
                tuple_counter -= 1
                domain[i][temp] = '0'
            perfect_pattern.append('0')
            idx += block_len + 1

        patterns = [''.join(perfect_pattern)]

    return tuple_counter, patterns


def revise_row(patterns, domain, n, idx):
    new_patterns = [
        pattern for pattern in patterns
        if all(
            not ((domain[idx][i] == '1' and pattern[i] == '0') or
                 domain[idx][i] == '0' and pattern[i] == '1')
                 for i in range(n)
        )
    ]
    return new_patterns


def revise_col(patterns, domain, n, idx):
    new_patterns = [
        pattern for pattern in patterns
        if all(
            not ((domain[i][idx] == '1' and pattern[i] == '0') or
                 domain[i][idx] == '0' and pattern[i] == '1')
                 for i in range(n)
        )
    ]
    return new_patterns


def overlap(patterns, domain, n, idx, tuple_counter, line_type):
    m = len(patterns)
    re_add_lst = []
    if line_type == 'C':
        for i in range(n):
            zero_one_map = defaultdict(int)
            for j in range(m):
                zero_one_map[patterns[j][i]] += 1
            
            if zero_one_map['0'] == 0:
                if domain[i][idx] == ('0', '1'):
                    tuple_counter -= 1
                    domain[i][idx] = '1'
                    re_add_lst.append((i, 'R'))
                
            elif zero_one_map['1'] == 0:
                if domain[i][idx] == ('0', '1'):
                    tuple_counter -= 1
                    domain[i][idx] = '0'
                    re_add_lst.append((i, 'R'))
    else:
        for i in range(n):
            zero_one_map = defaultdict(int)
            for j in range(m):
                zero_one_map[patterns[j][i]] += 1
            
            if zero_one_map['0'] == 0:
                if domain[idx][i] == ('0', '1'):
                    tuple_counter -= 1
                    domain[idx][i] = '1'
                    re_add_lst.append((i, 'C'))

            elif zero_one_map['1'] == 0:
                if domain[idx][i] == ('0', '1'):
                    tuple_counter -= 1
                    domain[idx][i] = '0'
                    re_add_lst.append((i, 'C'))

    return re_add_lst, tuple_counter


def backtrack_solution(domain, rows, cols, row_len, col_len, tuple_counter):
    def dfs(dom, rs, cs, tc):
        if tc == 0:
            return True, dom

        found = False
        for r in range(col_len):
            for c in range(row_len):
                if dom[r][c] == ('0','1'):
                    found = True
                    break
            if found:
                break

        if not found:
            return True, dom

        for val in ['0', '1']:
            dom_copy = [row[:] for row in dom]
            rs_copy = [p[:] for p in rs]
            cs_copy = [p[:] for p in cs]
            tc_copy = tc

            dom_copy[r][c] = val
            tc_copy -= 1

            queue = deque()
            queue.append((r, 'R'))
            queue.append((c, 'C'))

            contradiction = False
            while queue and not contradiction:
                idx, axis = queue.pop()
                if axis == 'R':
                    rs_copy[idx] = revise_row(rs_copy[idx], dom_copy, row_len, idx)
                    if not rs_copy[idx]:
                        contradiction = True
                        break
                    re_add, tc_copy = overlap(rs_copy[idx], dom_copy, row_len, idx, tc_copy, 'R')
                    for x in re_add:
                        queue.append(x)
                else:
                    cs_copy[idx] = revise_col(cs_copy[idx], dom_copy, col_len, idx)
                    if not cs_copy[idx]:
                        contradiction = True
                        break
                    re_add, tc_copy = overlap(cs_copy[idx], dom_copy, col_len, idx, tc_copy, 'C')
                    for x in re_add:
                        queue.append(x)

            if not contradiction:
                ok, final_dom = dfs(dom_copy, rs_copy, cs_copy, tc_copy)
                if ok:
                    return True, final_dom

        return False, None

    success, final_domain = dfs(domain, rows, cols, tuple_counter)
    if success:
        return final_domain
    return None


def save_result(domain):
    with open('zad_output.txt', 'w') as f:
        for l in domain:
            res_line = ''.join('#' if ch == '1' else '.' for ch in l)
            f.write(res_line + '\n')


if __name__ == '__main__':
    with open('zad_input.txt', 'r') as f:
        lines = f.readlines()
    
    col_len, row_len = map(int, lines[0].split())
    rows_spec = [list(map(int, line.strip().split())) for line
                 in lines[1:col_len + 1]]
    cols_spec = [list(map(int, line.strip().split())) for line
                 in lines[col_len + 1:col_len + 1 + row_len]]
    
    rows = []
    cols = []

    for spec in rows_spec:
        rows.append(generate_lines(spec, row_len))
    
    for spec in cols_spec:
        cols.append(generate_lines(spec, col_len))

    domain = [[('0', '1') for _ in range(row_len)] for _ in range(col_len)]
    tuple_counter = row_len * col_len
    # propagation queue
    queue = deque()

    # first check whether there is full line
    for idx, row in enumerate(rows_spec):
        tuple_counter, new_patterns = full_line(row, idx, row_len, domain, 'R', 
                                  tuple_counter, rows[idx])
        rows[idx] = new_patterns
        queue.append((idx, 'R'))

    for idx, col in enumerate(cols_spec):
        tuple_counter, new_patterns = full_line(col, idx, col_len, domain, 'C',
                                                 tuple_counter, cols[idx])
        cols[idx] = new_patterns
        queue.append((idx, 'C'))
    
    if tuple_counter == 0:
        save_result(domain)
        sys.exit(0)

    # now consider overlap
    while queue:
        idx, ax = queue.pop()
        # delete if something already doesn't match
        if ax == 'R':
            rows[idx] = revise_row(rows[idx], domain, row_len, idx)
            re_add_lst, tuple_counter = overlap(rows[idx], domain, 
                                                row_len, idx, tuple_counter, 'R')
        else:
            cols[idx] = revise_col(cols[idx], domain, col_len, idx)
            re_add_lst, tuple_counter = overlap(cols[idx], domain, 
                                                col_len, idx, tuple_counter, 'C')
        
        for tuple_ in re_add_lst:
            queue.append(tuple_)
        
    if tuple_counter == 0:
        save_result(domain)
        sys.exit(0)
    
    # domain = backtrack_solution(domain, rows, cols, row_len, col_len, tuple_counter)
    
    if domain:
        save_result(domain)
