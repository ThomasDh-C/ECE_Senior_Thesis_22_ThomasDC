import pandas as pd
import json

# excel = pd.ExcelFile('sdp_funcs.xls')
df = pd.read_excel('sdp funcs.xlsx', sheet_name='export')
names_info = {}
for r_idx in range(df.shape[0]):
    row = df.loc[r_idx, :]
    name, params = row.Name, row.Params
    name_info = {}
    # params looks like 'NVDLA_SDP_S_LUT_ADDR (0-9)'
    # but not necessarily with a hyphen
    for param in params.strip().split(', '):
        p_name, p_loc = param.split(' (')
        p_loc = p_loc[:-1]  # remove last )
        if '-' in p_loc:
            start, end = p_loc.split('-')
            start, end = int(start), int(end)
            name_info[p_name] = [start, end]
        else:
            start = int(p_loc)
            name_info[p_name] = [start, start]
    # 4 bytes (32 bits) for each csb register. Ignore the b part
    # ie should be 0xb000 not 0x000, but prev used 0x000 so use that
    padded_adr = hex(r_idx*4)[2:]
    padded_adr = '0x' + '0'*(3-len(padded_adr)) + padded_adr
    names_info[name] = {'addr': padded_adr, 'shifts': name_info}

with open('sdp_param_shifts_for_csb.json', 'w') as fout:
    # json.dump({'program fragment': self.prog_frag}, fout, indent=4)
    json.dump(names_info, fout, indent=4)

# to use
dictionary = {}
with open('sdp_param_shifts_for_csb.json', 'r') as fin:
    dictionary = json.load(fin)
