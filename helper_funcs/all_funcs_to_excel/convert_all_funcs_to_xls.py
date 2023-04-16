import pandas as pd
import json
import os
import re

df = pd.read_excel('all_regs.xlsx', sheet_name='input')

# Find all parameters for various CSB registers
full_names = list(df['Full_name'])
all_params = [[] for _ in range(len(full_names))]
full_names_not_found = set()
root_dir = "/root/nvdlahw/vmod/nvdla"
for core in os.listdir(root_dir):
    for data_file in os.listdir(f"{root_dir}/{core}"):
        with open(f"{root_dir}/{core}/{data_file}") as data_f:
            lines = data_f.readlines()
            for l_idx, l0 in enumerate(lines):
                if "// Register: " in l0 and "Field: " in l0 and l_idx + 2 < len(lines):
                    full_reg_name, field_name = re.findall(
                        r'// Register: ([^ ]+) +Field: ([^ ]+)', l0[:-1])[0]

                    # find correct row
                    reg_name_no_nvdla = full_reg_name
                    if reg_name_no_nvdla[:6] == 'NVDLA_':
                        reg_name_no_nvdla = reg_name_no_nvdla[6:]
                    if reg_name_no_nvdla[:4] == 'RBK_':
                        reg_name_no_nvdla = 'RUBIK_' + reg_name_no_nvdla[4:]
                    if reg_name_no_nvdla not in full_names:
                        # remove '_0' or '_1'
                        reg_name_no_nvdla = reg_name_no_nvdla[:-2]
                    if reg_name_no_nvdla == 'GLB_S_INTR_MASK':
                        reg_name_no_nvdla = 'GLB_INTR_MASK'
                    if reg_name_no_nvdla not in full_names:
                        full_names_not_found.add(reg_name_no_nvdla)
                        continue
                    correct_row = full_names.index(reg_name_no_nvdla)

                    # find correct shift info
                    l2 = lines[l_idx+2]
                    if 'reg_wr_data' not in l2:
                        print('Error')
                        raise Exception('No csb shifts - fix me')
                    shift_data_str = l2.split('reg_wr_data[')[1]
                    shift_data_str = shift_data_str.split(']')[0]
                    if ':' in shift_data_str:
                        end, start = shift_data_str.split(':')
                        all_params[correct_row].append(
                            (field_name, int(start), int(end)))
                    else:
                        all_params[correct_row].append(
                            (field_name, int(shift_data_str)))

print("Couldn't find register matches in excel for the following that appeared in the code:")
for name in sorted(full_names_not_found, key=lambda x: x.split('_')[0]):
    print(f'\t- {name}')
print('Note CYA does not appear anywhere in the HW arch docs')
print()

# Update the df and export an updated spreadsheet
print('The following registers did not have documentation in the HW files')
df['Params'] = ['' for _ in range(len(full_names))]
for r_idx, params in enumerate(all_params):
    params.sort(key=lambda x: x[1])
    stringified_params = []
    for param in params:
        field_name = 'NVDLA_' + df['Core'][r_idx] + '_' + param[0].upper()
        if len(param) == 2:
            p1 = param[1]
            stringified_params.append(f'{field_name} ({p1})')
        else:
            p1, p2 = param[1], param[2]
            stringified_params.append(f'{field_name} ({p1}-{p2})')
    if len(stringified_params) == 0:
        print('\t-'+df['Full_name'][r_idx] + " - " + df['Description'][r_idx])

    df['Params'][r_idx] = ', '.join(stringified_params)
df.to_excel('all_regs_with_params.xlsx')

# Export the json with all parameters
names_info = {}
for r_idx in range(df.shape[0]):
    row = df.loc[r_idx, :]
    name, address, params = row['Full_name'], row['Address'], row.Params

    # continue if couldn't find information on that register's parameters
    if len(params) == 0:
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

    names_info[name] = {'addr': address, 'shifts': name_info}

with open('all_param_shifts_for_csb.json', 'w') as fout:
    json.dump(names_info, fout, indent=4)

# to use
dictionary = {}
with open('all_param_shifts_for_csb.json', 'r') as fin:
    dictionary = json.load(fin)
