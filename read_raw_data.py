"""
读取 data/raw 目录中的原始数据文件
"""
import pandas as pd
import os

def read_excel_file(filepath):
    """读取 Excel 文件并显示详细信息"""
    print("=" * 80)
    print(f"文件: {os.path.basename(filepath)}")
    print("=" * 80)
    
    # 读取所有工作表
    excel_file = pd.ExcelFile(filepath)
    print(f"\n工作表数量: {len(excel_file.sheet_names)}")
    print(f"工作表名称: {excel_file.sheet_names}")
    
    for sheet_name in excel_file.sheet_names:
        print("\n" + "-" * 80)
        print(f"工作表: {sheet_name}")
        print("-" * 80)
        
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        # 基本信息
        print(f"\n数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"\n列名:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        # 数据类型
        print(f"\n数据类型:")
        print(df.dtypes)
        
        # 缺失值统计
        print(f"\n缺失值统计:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  无缺失值")
        
        # 显示前几行数据
        print(f"\n前 10 行数据:")
        print(df.head(10))
        
        # 显示最后几行数据
        print(f"\n后 5 行数据:")
        print(df.tail(5))
        
        # 数值列的基本统计信息
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print(f"\n数值列统计信息:")
            print(df[numeric_cols].describe())
        
        # 文本列的唯一值数量
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            print(f"\n文本列唯一值数量:")
            for col in text_cols:
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count} 个唯一值")
                if unique_count <= 20:
                    print(f"    唯一值: {df[col].unique()[:20].tolist()}")
        
        print("\n")

if __name__ == "__main__":
    # 读取 data/raw 目录中的所有 Excel 文件
    raw_dir = "data/raw"
    
    if not os.path.exists(raw_dir):
        print(f"错误: 目录 {raw_dir} 不存在")
    else:
        excel_files = [f for f in os.listdir(raw_dir) if f.endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            print(f"在 {raw_dir} 目录中未找到 Excel 文件")
        else:
            print(f"找到 {len(excel_files)} 个 Excel 文件\n")
            
            for excel_file in excel_files:
                filepath = os.path.join(raw_dir, excel_file)
                read_excel_file(filepath)

