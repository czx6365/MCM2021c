"""
展示原始数据的样本
"""
import pandas as pd
import sys

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 100)
print("主数据集样本 (2021MCMProblemC_DataSet.xlsx)")
print("=" * 100)
df1 = pd.read_excel('data/raw/2021MCMProblemC_DataSet.xlsx')
print("\n前 15 条记录的关键字段:")
print(df1[['GlobalID', 'Detection Date', 'Lab Status', 'Latitude', 'Longitude']].head(15))
print(f"\n总记录数: {len(df1)}")

print("\n\n" + "=" * 100)
print("Lab Status 分布:")
print("=" * 100)
print(df1['Lab Status'].value_counts())

print("\n\n" + "=" * 100)
print("图像关联表样本 (2021MCM_ProblemC_ Images_by_GlobalID.xlsx)")
print("=" * 100)
df2 = pd.read_excel('data/raw/2021MCM_ProblemC_ Images_by_GlobalID.xlsx')
print("\n前 15 条记录:")
print(df2.head(15))
print(f"\n总记录数: {len(df2)}")

print("\n\n" + "=" * 100)
print("文件类型分布:")
print("=" * 100)
print(df2['FileType'].value_counts())

print("\n\n" + "=" * 100)
print("数据关联统计:")
print("=" * 100)
print(f"主数据集中的 GlobalID 数量: {df1['GlobalID'].nunique()}")
print(f"图像关联表中的 GlobalID 数量: {df2['GlobalID'].nunique()}")
print(f"图像关联表中的文件数量: {len(df2)}")
print(f"平均每个 GlobalID 的文件数: {len(df2) / df2['GlobalID'].nunique():.2f}")

# 检查有多少主数据记录有对应的图像
common_ids = set(df1['GlobalID'].str.strip('{}')) & set(df2['GlobalID'].str.strip('{}'))
print(f"\n有图像文件的主数据记录数: {len(common_ids)}")
print(f"无图像文件的主数据记录数: {df1['GlobalID'].nunique() - len(common_ids)}")

