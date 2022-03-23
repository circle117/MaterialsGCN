from rdkit import Chem
import pandas as pd
import os

DATASET_PATH = './Dataset_test'
EXCEL_NAME = 'data.xlsx'
TG_METHOD = 'ALL'               # ALL or method's name
IF_TYPE = False                  # use TYPE as feature or not
METHOD = 2                      # 0 for all, 1 for one-step, 2 for two-step
SOLVENT = ['DMAc', 'NMP']
TIME1_CON = True                # True for continuous, Flase for disperse
OVERNIGHT = 20
SEVERAL = 6
RT = 20
TEMPERATURE_DIC = {
    0: '0-14',
    5: '0-14',
    20: '15-25',
    25: '15-25',
    30: '25-35',
    70: '65-75',
    100: '95-105',
    '0-5': '0-14',
    '15-25': '15-25',
    '20-25': '15-25',
    'RT': '15-25'
}
TIME1_DIC = {
    'overnight': '16-24',
    16: '16-24',
    17: '16-24',
    20: '16-24',
    24: '16-24',
    48: '48',
    10: '10-15',
    12: '10-15',
    15: '10-15',
    4: '4-10',
    5: '4-10',
    6: '4-10',
    7: '4-10',
    8: '4-10',
    1: '1-3',
    1.5: '1-3',
    2: '1-3',
    2.5: '1-3',
    3: '1-3'
}

CSV_NAME = 'data_Method%d.csv' % METHOD


def get_smiles(pid):
    mol = Chem.MolFromMolFile(os.path.join(DATASET_PATH, 'Mol', pid+'.mol'))
    return Chem.MolToSmiles(mol)


def get_time_1(time):
    if time == 'overnight':
        return OVERNIGHT
    else:
        return round(time)


def get_time_2(time):
    if time == 'overnight':
        return OVERNIGHT
    elif time == 'several':
        return SEVERAL
    else:
        return round(time)


def get_temperature(temp):
    if temp=='RT':
        return RT
    else:
        return temp

df = pd.read_excel(os.path.join(DATASET_PATH, EXCEL_NAME))

# mol文件转SMILES
df['SMILES'] = df.apply(lambda x: get_smiles(x['PID']), axis=1)
df.drop('PID', axis=1, inplace=True)

# 玻璃化转变温度
if TG_METHOD != 'ALL':
    df = df[df['Tg_Method'] == TG_METHOD]
df.drop('Tg_Method', axis=1, inplace=True)

# 是否使用反应类型作为特征
if not IF_TYPE:
    df.drop('Type', axis=1, inplace=True)
else:
    df = df[(df['Type'] == 'Polyaddition') | (df['Type'] == 'Polycondensation')]

# 聚酰亚胺生成方法
if METHOD == 0:                     # 所有生成方法，用于训练GCN
    df = df.sample(frac=1).reset_index(drop=True)               # shuffle
    order = ['SMILES', 'Tg']
    df = df[order]
    df.to_csv(os.path.join(DATASET_PATH, CSV_NAME), encoding='utf8', index=False)
    exit()
else:
    df = df[df['Method'] == METHOD]

# 反应溶剂
df = df[(df['Solvent'] == SOLVENT[0]) | (df['Solvent'] == SOLVENT[1])]

# 反应温度
df['Temperature1'] = df['Temperature1'].apply(TEMPERATURE_DIC.get)

# 反应时间
if TIME1_CON:
    df['Time1'] = df.apply(lambda x: get_time_1(x['Time1']), axis=1)
else:
    df['Time1'] = df['Time1'].apply(TIME1_DIC.get)

# 处理第二步
if METHOD == 2:
    # 反应时间
    df['Time2'] = df.apply(lambda x:get_time_2(x['Time2']), axis=1)

    # 反应温度
    df['min_temp'] = df.apply(lambda x:get_temperature(x['min_temp']), axis=1)
    df['max_temp'] = df.apply(lambda x:get_temperature(x['max_temp']), axis=1)

# 保存
if METHOD == 1:
    pass
elif METHOD == 2:
    df.drop('Method', axis=1, inplace=True)
    order = ['SMILES', 'Solvent', 'Temperature1', 'Time1',
             'Method2', 'min_temp', 'max_temp', 'Time2', 'Tg']
    df = df[order]

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(os.path.join(DATASET_PATH, CSV_NAME), encoding='utf8', index=False)