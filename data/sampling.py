import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def sample_and_split_per_category(input_csv, sample_size, output_dir, random_state=95):
    df = pd.read_csv(input_csv)
    sampled_list = []
    # 分组采样
    for category, group in df.groupby('new_category'):
        if len(group) >= sample_size:
            sampled = group.sample(n=sample_size, random_state=random_state)
            sampled_list.append(sampled)
        else:
            print("[WARN] 类别 {} 样本 {} 少于 {}，已跳过。".format(category, len(group), sample_size))
    if not sampled_list:
        raise ValueError("未找到任何满足采样数量的类别。")
    sampled_df = pd.concat(sampled_list).reset_index(drop=True)
    # 分层划分
    train_df, test_df = train_test_split(sampled_df, test_size=0.2, stratify=sampled_df['new_category'], random_state=random_state)
    # 准备输出
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_sample_{}.csv".format(sample_size)
    test_out = out_dir / "test_sample_{}.csv".format(sample_size)
    # 重命名列并保存
    train_df[['new_category', 'instance_path']].rename(columns={'new_category': 'category'}).to_csv(train_out, index=False, encoding='utf-8-sig')
    test_df[['new_category', 'instance_path']].rename(columns={'new_category': 'category'}).to_csv(test_out, index=False, encoding='utf-8-sig')
    print("[INFO] 已保存训练集: {}".format(train_out))
    print("[INFO] 已保存测试集: {}".format(test_out))


if __name__ == "__main__":
    # 配置
    component_index_path = r"Q:\pychem_project\BIMCompNet\data\relabel_component_index.csv"
    sample_output_dir = r"Q:\pychem_project\BIMCompNet\data"
    sizes = [5000, 1000, 500, 100]

    # 批量生成
    for size in sizes:
        sample_and_split_per_category(component_index_path, size, sample_output_dir)
