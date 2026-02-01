
import platform
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. 全局配置 (Configuration)
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "UserBehavior.csv"
OUTPUT_DIR = BASE_DIR / "Item_png"  # 结果保存路径

# 算法超参数
TOP_K_NEIGHBORS = 20        # 选取Top-K相似物品
TOP_N_RECOMMEND = 5         # 为每个用户推荐N个物品
MAX_ITEMS_FOR_SIM = 2000    # 限制物品数量，防止计算量过大 (Item-Based 核心)
HEATMAP_ITEM_SAMPLE = 30    # 热力图展示的物品数量

# 设置中文字体 (SimHei)
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 全局变量，用于存储字体属性
GLOBAL_FONT_PROP = None

# 定义SCI风格配色方案 (Warm/Vibrant Style - Distinct from User-Based)
# 采用红/紫/橙暖色调为主，避免与 User-Based 的蓝/冷色调撞车
SCI_COLORS = [
    "#D62728", # 0. Brick Red
    "#9467BD", # 1. Muted Purple
    "#8C564B", # 2. Chestnut Brown
    "#E377C2", # 3. Raspberry Yogurt Pink
    "#7F7F7F", # 4. Middle Gray
    "#BCBD22", # 5. Curry Yellow-Green
    "#17BECF", # 6. Blue-Teal
    "#FF7F0E", # 7. Safety Orange
    "#2CA02C", # 8. Cooked Asparagus Green
    "#1F77B4"  # 9. Muted Blue
]

def setup_matplotlib_style() -> None:
    """
    配置Matplotlib绘图风格
    1. 强制加载 Windows 系统自带的中文字体文件，解决中文乱码。
    2. 设置SCI期刊风格的配色和图表样式。
    """
    global GLOBAL_FONT_PROP
    
    font_path = ""
    system = platform.system()
    
    if system == "Windows":
        candidates = [
            r"C:\Windows\Fonts\simhei.ttf",  # 黑体
            r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
            r"C:\Windows\Fonts\simsun.ttc",  # 宋体
        ]
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break
    
    if font_path:
        try:
            # 记录全局字体属性，后续所有 text, title, label 显式使用它
            GLOBAL_FONT_PROP = fm.FontProperties(fname=font_path)
            # 同时尝试注册到全局
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = GLOBAL_FONT_PROP.get_name()
            plt.rcParams["font.sans-serif"] = [GLOBAL_FONT_PROP.get_name()]
            print(f"已成功加载字体: {GLOBAL_FONT_PROP.get_name()}")
        except Exception as e:
            print(f"字体加载失败: {e}")
    
    # 设置通用的绘图参数
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.5,
    })

# ==========================================
# 1. 数据加载与预处理 (Data Loading & Preprocessing)
# ==========================================

def load_data(file_path: Path) -> pd.DataFrame:
    """
    读取原始数据文件并统一字段名。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
        
    df = pd.read_csv(file_path)
    expected_columns = [
        "user_id", "item_id", "brand", "brand_id", "item_name",
        "item_category", "item_category_id", "behavior_type", "timestamp", "price"
    ]
    if df.shape[1] == 10 and list(df.columns) != expected_columns:
        df.columns = expected_columns
    return df

def normalize_behavior_type(series: pd.Series) -> pd.Series:
    """
    将行为类型统一为中文标签。
    """
    mapping_text = {"pv": "浏览", "buy": "购买", "cart": "加购", "fav": "收藏", "view": "浏览", "purchase": "购买"}
    mapping_cn = {"浏览": "浏览", "购买": "购买", "加购": "加购", "收藏": "收藏"}
    
    series_str = series.astype(str).str.lower()
    mapped = series_str.map(mapping_text)
    mapped = mapped.fillna(series.astype(str).map(mapping_cn))
    return mapped.fillna(series.astype(str))

def behavior_score(series: pd.Series) -> pd.Series:
    """
    将行为类型映射为数值强度 (Implicit Feedback -> Explicit Score)。
    """
    mapping = {
        "浏览": 1.0, "收藏": 2.0, "加购": 3.0, "购买": 4.0,
        "pv": 1.0, "fav": 2.0, "cart": 3.0, "buy": 4.0
    }
    numeric = pd.to_numeric(series, errors="coerce")
    text_score = series.astype(str).str.lower().map(mapping)
    score = text_score.fillna(numeric).fillna(1.0)
    return score.astype(float)

def convert_timestamp(series: pd.Series) -> pd.Series:
    """
    将时间戳转换为 datetime 对象。
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.to_datetime(series, errors="coerce")
    max_value = numeric.max()
    unit = "ms" if max_value > 1e12 else "s"
    return pd.to_datetime(numeric, unit=unit, errors="coerce")

def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造用户-商品交互矩阵 (User-Item Matrix)。
    行: user_id, 列: item_id, 值: 行为得分总和
    """
    matrix = df.pivot_table(
        index="user_id",
        columns="item_id",
        values="behavior_score",
        aggfunc="sum",
        fill_value=0.0,
    )
    return matrix

def sample_popular_items(matrix: pd.DataFrame, max_items: int) -> pd.DataFrame:
    """
    【Item-Based 特有】
    由于商品数量通常极其庞大，计算全量 Item-Item 相似度矩阵 (N*N) 内存会爆炸。
    这里选取“最热门”（交互用户最多）的 Top-N 商品进行算法演示。
    """
    if matrix.shape[1] <= max_items:
        return matrix
    
    # 计算每个商品的交互总次数（或总分）
    item_popularity = matrix.sum(axis=0)
    # 选取 Top-N 热门商品
    top_items = item_popularity.nlargest(max_items).index
    
    # 过滤矩阵，只保留热门商品
    # 注意：这会导致只推荐这些热门商品，属于长尾截断
    sampled_matrix = matrix.loc[:, top_items]
    
    # 同时也过滤掉没有任何交互的用户（如果存在）
    sampled_matrix = sampled_matrix.loc[(sampled_matrix != 0).any(axis=1)]
    
    print(f"已截取 Top-{max_items} 热门商品进行建模，矩阵形状: {sampled_matrix.shape}")
    return sampled_matrix

# ==========================================
# 2. 核心算法: 基于物品的协同过滤 (Item-Based CF)
# ==========================================

def compute_item_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    计算物品之间的相似度 (Item-Item Similarity)。
    原理：喜欢物品 A 的用户，是否也喜欢物品 B？
    """
    # Item-Based 需要计算列向量（Item Vectors）之间的相似度
    # matrix.T 转置后，行变为 Item，列变为 User
    item_matrix = matrix.T
    
    # 计算余弦相似度
    similarity = cosine_similarity(item_matrix.values)
    similarity_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
    return similarity_df

def generate_item_based_recommendations(
    matrix: pd.DataFrame,
    item_similarity_df: pd.DataFrame,
    top_k: int,
    top_n: int,
) -> pd.DataFrame:
    """
    基于物品相似度生成推荐。
    公式：Score(u, i) = Σ ( Sim(i, j) * Rating(u, j) )
    其中 j 是用户 u 历史交互过的物品。
    """
    recommendations = []
    
    # 遍历每个用户
    # 注意：这里为了代码可读性使用了循环，大规模数据应使用矩阵乘法加速
    # Matrix Multiplication Approach: Pred = User_Matrix @ Item_Sim_Matrix
    
    # 为了演示 Top-K Neighbor 的逻辑，我们采用一种混合方式：
    # 1. 用户的历史交互向量
    # 2. 对每个历史物品，找到其相似物品
    
    print("正在生成推荐列表 (Item-Based)...")
    
    # 使用矩阵运算加速：Scores = User_Matrix x Item_Similarity
    # 这里的 item_similarity_df 是完整的，如果只取 Top-K，需要先把非 Top-K 置为 0
    
    # 1. 过滤相似度矩阵，只保留每行 Top-K 的值 (Hard Thresholding)
    sim_values = item_similarity_df.values.copy()
    # 对每一行（每个 Target Item），只保留 Top-K 个相似 items
    # 这里的行 i 代表 Target Item i，列 j 代表 Neighbor Item j
    # 我们希望用 Top-K 相似的 j 来预测 i
    
    # 由于是 Item-Based，预测评分 r_ui = sum(r_uj * sim_ij) / sum(sim_ij)
    # 我们可以直接矩阵相乘
    
    # 预处理：保留 Top-K
    for i in range(sim_values.shape[0]):
        # 找到第 K 大的值
        row = sim_values[i, :]
        # argpartition 会把第 K 大的元素放到位置上，右边是比它大的
        # 我们要保留最大的 K 个，其他的置 0
        if len(row) > top_k:
            threshold = np.partition(row, -top_k)[-top_k]
            sim_values[i, :][row < threshold] = 0
            
    # 2. 矩阵乘法计算得分
    # matrix (U x I) dot sim_values (I x I) -> scores (U x I)
    # 注意 sim_values 是对称的（如果不截断），但截断后可能不对称
    predicted_scores = matrix.values.dot(sim_values)
    
    # 3. 归一化（可选，这里简化处理，直接用加权和）
    
    # 4. 生成 Top-N
    users = matrix.index
    items = matrix.columns
    
    for idx, user_id in enumerate(users):
        user_scores = predicted_scores[idx]
        user_history = matrix.values[idx]
        
        # 过滤已交互物品：将历史交互过的物品分数设为负无穷
        user_scores[user_history > 0] = -np.inf
        
        # 获取 Top-N 索引
        # argpartition 效率更高
        if len(user_scores) > top_n:
            top_indices = np.argpartition(user_scores, -top_n)[-top_n:]
            # 还需要按分数排序
            top_indices = top_indices[np.argsort(user_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(user_scores)[::-1]
            
        for rank, item_idx in enumerate(top_indices, start=1):
            item_id = items[item_idx]
            score = user_scores[item_idx]
            
            # 只有分数为正才推荐
            if score > -1e9: # 考虑到负无穷
                recommendations.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rank": rank,
                    "score": score
                })
                
    return pd.DataFrame(recommendations)

def attach_item_info(recs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    关联商品详细信息。
    """
    item_info = (
        df.drop_duplicates(subset=["item_id"])
        .loc[:, ["item_id", "item_name", "item_category", "brand", "price"]]
        .set_index("item_id")
    )
    return recs.join(item_info, on="item_id")

def save_recommendations(recs: pd.DataFrame, output_dir: Path) -> None:
    """
    保存推荐结果。
    """
    output_path = output_dir / "物品协同过滤推荐结果.csv"
    recs.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"推荐结果已保存至: {output_path}")

# ==========================================
# 3. 评估模块 (Evaluation)
# ==========================================

def evaluate_algorithm(matrix: pd.DataFrame, item_similarity_df: pd.DataFrame, top_n: int) -> None:
    """
    评估 Item-Based CF 的 Precision, Recall, Coverage。
    基于训练集拟合度进行评估。
    """
    hits = 0
    total_relevant = 0
    recommended_count = 0
    all_recommended_items = set()
    
    # 随机采样用户进行评估
    eval_users = matrix.sample(min(500, matrix.shape[0]), random_state=42).index
    
    # 准备相似度矩阵 (Top-K Filtered)
    sim_values = item_similarity_df.values.copy()
    for i in range(sim_values.shape[0]):
        row = sim_values[i, :]
        if len(row) > TOP_K_NEIGHBORS:
            threshold = np.partition(row, -TOP_K_NEIGHBORS)[-TOP_K_NEIGHBORS]
            sim_values[i, :][row < threshold] = 0
            
    # 批量预测
    # 仅计算评估用户的分数
    user_indices = [matrix.index.get_loc(u) for u in eval_users]
    sub_matrix = matrix.iloc[user_indices].values
    pred_scores = sub_matrix.dot(sim_values)
    
    for i, user_id in enumerate(eval_users):
        ground_truth = sub_matrix[i] > 0
        true_item_indices = np.where(ground_truth)[0]
        
        if len(true_item_indices) == 0:
            continue
            
        # 预测分数
        scores = pred_scores[i]
        
        # Top-N (包括已交互的，为了计算 Recall)
        # 注意：这里我们不 Mask 掉已交互的，因为是 Training Set Evaluation
        # 我们看算法是否给用户实际上喜欢的物品打了高分
        
        if len(scores) > top_n:
            top_indices = np.argpartition(scores, -top_n)[-top_n:]
        else:
            top_indices = np.argsort(scores)
            
        # 统计命中
        hit_count = len(set(top_indices) & set(true_item_indices))
        hits += hit_count
        total_relevant += len(true_item_indices)
        recommended_count += top_n
        all_recommended_items.update(matrix.columns[top_indices])
        
    precision = hits / recommended_count if recommended_count > 0 else 0.0
    recall = hits / total_relevant if total_relevant > 0 else 0.0
    coverage = len(all_recommended_items) / matrix.shape[1]
    
    print("-" * 40)
    print(f"Item-Based 算法评估结果 (Top-{top_n}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Coverage:  {coverage:.4f}")
    print("-" * 40)

# ==========================================
# 4. 可视化模块 (Visualization)
# ==========================================
# 注：基础 EDA 图表函数与 User-Based 类似，但保存路径不同

def plot_basic_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制基础数据分析图表 (EDA)，复用 User-Based 的逻辑但保存到新目录。
    """
    # 1. 用户行为类型分布
    plt.figure(figsize=(10, 6))
    counts = df["behavior_label"].value_counts()
    palette = SCI_COLORS[:len(counts)]
    ax = sns.barplot(x=counts.index, y=counts.values, palette=palette)
    ax.bar_label(ax.containers[0], padding=3, fontsize=12)
    plt.title("用户行为类型分布", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("行为类型", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("次数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    for t in ax.get_xticklabels(): t.set_fontproperties(GLOBAL_FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_dir / "用户行为类型分布.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 商品类别Top10
    top_categories = df["item_category"].value_counts().head(10)
    plt.figure(figsize=(12, 7))
    # 使用 rocket 调色盘，与 User-Based 的 mako 区分
    ax = sns.barplot(x=top_categories.values, y=top_categories.index.astype(str), palette="rocket")
    ax.bar_label(ax.containers[0], padding=3, fontsize=11)
    plt.title("商品类别热度 Top10", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("交互次数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("商品类别 ID", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    for t in ax.get_yticklabels() + ax.get_xticklabels(): t.set_fontproperties(GLOBAL_FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_dir / "商品类别Top10.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. 商品价格分布
    plt.figure(figsize=(10, 6))
    price_data = df["price"].dropna()
    limit = price_data.quantile(0.99)
    filtered_price = price_data[price_data <= limit]
    sns.histplot(filtered_price, bins=50, kde=True, color=SCI_COLORS[0], alpha=0.6, edgecolor="white")
    plt.title("商品价格分布 (Top 99%)", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("价格", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("频数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_dir / "商品价格分布.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. 用户活跃度分布
    user_counts = df["user_id"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_counts, bins=50, kde=True, color=SCI_COLORS[4], alpha=0.6, edgecolor="white")
    plt.yscale("log")
    plt.title("用户活跃度分布 (对数坐标)", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("用户交互次数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("用户数 (Log Scale)", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    ax = plt.gca()
    for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_fontproperties(GLOBAL_FONT_PROP)
    plt.tight_layout()
    plt.savefig(output_dir / "用户活跃度分布.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. 用户行为小时分布
    if not df["datetime"].isna().all():
        hourly_counts = df["datetime"].dt.hour.value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        plt.plot(hourly_counts.index, hourly_counts.values, marker="o", color=SCI_COLORS[3], linewidth=2.5)
        plt.fill_between(hourly_counts.index, hourly_counts.values, color=SCI_COLORS[3], alpha=0.15)
        plt.title("用户活跃时段分布 (24小时)", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
        plt.xlabel("小时 (0-23)", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
        plt.ylabel("交互次数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
        plt.xticks(range(0, 24))
        ax = plt.gca()
        for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_fontproperties(GLOBAL_FONT_PROP)
        plt.tight_layout()
        plt.savefig(output_dir / "用户行为小时分布.png", dpi=300, bbox_inches="tight")
        plt.close()

def plot_item_similarity_heatmap(similarity_df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制物品相似度热力图 (Item-Item Similarity Heatmap)
    """
    if similarity_df.shape[0] < 2:
        return

    sample_count = min(HEATMAP_ITEM_SAMPLE, similarity_df.shape[0])
    # 随机采样一部分物品进行展示
    sample_items = similarity_df.sample(sample_count, random_state=42).index
    sample_matrix = similarity_df.loc[sample_items, sample_items]

    plt.figure(figsize=(10, 9))
    ax = sns.heatmap(
        sample_matrix,
        cmap="viridis", # Item-Based 换个色系，用 Viridis (黄绿蓝) 区分
        center=0,
        annot=False,
        square=True,
        cbar_kws={"label": "余弦相似度"}
    )
    
    try:
        cbar = ax.collections[0].colorbar
        cbar.set_label("余弦相似度", fontproperties=GLOBAL_FONT_PROP)
        cbar.ax.yaxis.set_tick_params(labelsize=10) 
    except Exception:
        pass

    plt.title(f"物品相似度矩阵热力图 (随机采样 {sample_count} 物品)", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("物品 ID", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("物品 ID", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)

    plt.tight_layout()
    plt.savefig(output_dir / "物品相似度热力图.png", dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================
# 5. 主程序 (Main)
# ==========================================

def main() -> None:
    print(">>> 启动 Item-Based Collaborative Filtering 算法程序...")
    
    setup_matplotlib_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    print("正在读取数据...")
    df = load_data(DATA_PATH)
    
    # 2. 预处理
    print("正在进行数据预处理...")
    df["behavior_label"] = normalize_behavior_type(df["behavior_type"])
    df["behavior_score"] = behavior_score(df["behavior_type"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["datetime"] = convert_timestamp(df["timestamp"])
    
    # 3. 基础可视化
    print("正在生成基础分析图表...")
    plot_basic_eda(df, OUTPUT_DIR)
    
    # 4. 构建矩阵与建模
    print("正在构建用户-物品矩阵...")
    user_item_matrix = build_user_item_matrix(df)
    print(f"原始矩阵大小: {user_item_matrix.shape}")
    
    # 截取热门物品 (Item-Based 关键步骤)
    # 为了保证计算 Item-Item 相似度矩阵 (Items x Items) 速度，必须限制物品数
    sampled_matrix = sample_popular_items(user_item_matrix, MAX_ITEMS_FOR_SIM)
    
    print("正在计算物品相似度矩阵...")
    item_similarity_df = compute_item_similarity(sampled_matrix)
    
    # 绘制物品相似度热力图
    plot_item_similarity_heatmap(item_similarity_df, OUTPUT_DIR)
    
    # 5. 生成推荐
    recs = generate_item_based_recommendations(
        sampled_matrix,
        item_similarity_df,
        TOP_K_NEIGHBORS,
        TOP_N_RECOMMEND
    )
    
    if not recs.empty:
        recs = attach_item_info(recs, df)
        save_recommendations(recs, OUTPUT_DIR)
    else:
        print("警告: 未生成任何推荐结果，可能是数据过少或过滤过严。")
        
    # 6. 算法评估
    evaluate_algorithm(sampled_matrix, item_similarity_df, TOP_N_RECOMMEND)
    
    print("\n>>> 所有任务已完成！请查看 Item_png 目录。")

if __name__ == "__main__":
    main()
