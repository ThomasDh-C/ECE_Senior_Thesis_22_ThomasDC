import re
lines = []
with open('sdp.py', 'r') as f:
    lines = f.readlines()

data = ''.join(lines)
ir_strings = [r[2] for r in re.findall(
    '\{(.|\n)*?name\': f?(\'|\")((.|\n)*?\}\))', data)]
ir_strings = [s.replace('\"', '\'') for s in ir_strings]
ir_strings = [s.replace('f\'', '\'') for s in ir_strings]
ir_data = []
for s in ir_strings:
    row = ['', []]
    s_ls = [a.strip() for a in s.split('\n')]
    row[0] = s_ls[0].split('\'')[0]
    for sl in s_ls[1:]:
        if len(sl) > 2:
            row[1].append(sl.split('\'')[1])
    ir_data.append(row)

with open('all_poss_acceleratorIR.txt', 'w') as w:
    for d in ir_data:
        params = ', '.join(d[1])
        w.write(f'{d[0]}: {params}\n')
