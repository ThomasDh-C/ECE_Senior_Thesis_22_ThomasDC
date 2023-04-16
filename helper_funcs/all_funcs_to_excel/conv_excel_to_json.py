import pandas as pd
import json

df = pd.read_excel('all_regs_with_params_2.xlsx')
names_info = {}
for r_idx in range(df.shape[0]):
    row = df.loc[r_idx, :]
    name, params = row.Full_name, row.Params
    if pd.isna(row.Params):
        continue
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
    padded_adr = row.Address
    names_info[name] = {'addr': padded_adr, 'shifts': name_info}

with open('all_param_shifts_for_csb.json', 'w') as fout:
    json.dump(names_info, fout, indent=4)
