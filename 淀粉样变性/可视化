import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# ============================================================
# 路径配置（只改这两行）
# ============================================================
data_path = r"D:\aaaSCNU\0data\Amyloid王惊华\代谢组学\数据_李梦圆\ALHDPB代谢物统计结果.xlsx"
save_dir  = r"D:\aaaSCNU\0data\Amyloid王惊华\代谢组学"

# 自动提取文件名前缀
# 如果文件名很长只想取前一部分，改max_len的数字

basename = os.path.splitext(os.path.basename(data_path))[0].split('代谢物统计结果')[0]
print(f"文件名前缀：{basename}")

# ============================================================
# 读取数据 + 筛差异代谢物
# ============================================================
df = pd.read_excel(data_path, sheet_name=0)
df = df.rename(columns={'Unnamed: 0': 'MetaboliteName'})
df = df.dropna(subset=['FC', 'p.ajusted', 'VIP'])
df['log2FC'] = np.log2(df['FC'])
df['-log10p'] = -np.log10(df['p.ajusted'])

up   = (df['log2FC'] >  1) & (df['p.ajusted'] < 0.05) & (df['VIP'] > 1)
down = (df['log2FC'] < -1) & (df['p.ajusted'] < 0.05) & (df['VIP'] > 1)
ns   = ~(up | down)

df_diff = df[up | down].copy()
df_diff['direction'] = np.where(df_diff['log2FC'] > 0, '上调', '下调')

print(f"上调：{up.sum()}  下调：{down.sum()}  不显著：{ns.sum()}")

# ============================================================
# 图1：火山图
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(df.loc[ns,   'log2FC'], df.loc[ns,   '-log10p'],
           c='#B4B2A9', s=15, alpha=0.5, linewidths=0, label='Not significant')
ax.scatter(df.loc[up,   'log2FC'], df.loc[up,   '-log10p'],
           c='#E24B4A', s=20, alpha=0.8, linewidths=0, label=f'Up ({up.sum()})')
ax.scatter(df.loc[down, 'log2FC'], df.loc[down, '-log10p'],
           c='#378ADD', s=20, alpha=0.8, linewidths=0, label=f'Down ({down.sum()})')

ax.axvline(x= 1, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axvline(x=-1, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

ax.set_xlabel('log₂(Fold Change)', fontsize=12)
ax.set_ylabel('-log₁₀(p adjusted)', fontsize=12)
ax.set_title(f'Volcano Plot - {basename}', fontsize=13)
ax.legend(fontsize=10, framealpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{basename}_火山图.svg'), format='svg')
plt.show()
print("火山图已保存")

# ============================================================
# 图2：ClassI 分布图
# ============================================================
count1 = df_diff.groupby(['ClassI (Chinese)', 'direction']).size().unstack(fill_value=0)
for col in ['上调', '下调']:
    if col not in count1.columns:
        count1[col] = 0
count1['total'] = count1['上调'] + count1['下调']
count1 = count1.sort_values('total', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(count1.index, count1['上调'],  color='#E24B4A', label='上调', alpha=0.85)
ax.barh(count1.index, -count1['下调'], color='#378ADD', label='下调', alpha=0.85)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('代谢物数量', fontsize=12)
ax.set_title(f'差异代谢物 ClassI 分布 - {basename}', fontsize=13)
ax.legend(fontsize=10)
xticks = ax.get_xticks()
ax.set_xticklabels([str(int(abs(x))) for x in xticks])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{basename}_ClassI分布.svg'), format='svg')
plt.show()
print("ClassI分布图已保存")


# ============================================================
# 输出差异代谢物列表
# ============================================================
df_diff_output = df[up | down].copy()
df_diff_output['direction'] = np.where(df_diff_output['log2FC'] > 0, 'Up', 'Down')

# 选择需要的列输出
cols = ['MetaboliteName', 'log2FC', 'p.ajusted', 'VIP', 'direction',
        'HMDB_ID', 'KEGG_ID', 'ClassI (Chinese)', 'ClassII (Chinese)']

# 只保留存在的列（有些列可能为空）
cols_exist = [c for c in cols if c in df_diff_output.columns]
df_diff_output = df_diff_output[cols_exist]

# 保存
output_path = os.path.join(save_dir, f'{basename}_差异代谢物.xlsx')
df_diff_output.to_excel(output_path, index=False)
print(f"差异代谢物已保存：{output_path}")
print(f"共 {len(df_diff_output)} 个，上调 {(df_diff_output['direction']=='Up').sum()} 个，下调 {(df_diff_output['direction']=='Down').sum()} 个")

# ============================================================
# 图3：ClassII 脂质亚类分布
# ============================================================
df_lipid = df_diff[df_diff['ClassI (Chinese)'] == '脂质和类脂分子'].copy()
count2 = df_lipid.groupby(['ClassII (Chinese)', 'direction']).size().unstack(fill_value=0)
for col in ['上调', '下调']:
    if col not in count2.columns:
        count2[col] = 0
count2['total'] = count2['上调'] + count2['下调']
count2 = count2.sort_values('total', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(count2.index, count2['上调'],  color='#E24B4A', label='上调', alpha=0.85)
ax.barh(count2.index, -count2['下调'], color='#378ADD', label='下调', alpha=0.85)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('代谢物数量', fontsize=12)
ax.set_title(f'脂质亚类分布 ClassII - {basename}', fontsize=13)
ax.legend(fontsize=10)
xticks = ax.get_xticks()
ax.set_xticklabels([str(int(abs(x))) for x in xticks])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{basename}_ClassII脂质分布.svg'), format='svg')
plt.show()
print("ClassII脂质分布图已保存")