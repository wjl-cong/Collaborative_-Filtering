from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "UserBehavior.csv"
OUTPUT_DIR = BASE_DIR / "User_png"

TOP_K_NEIGHBORS = 20
TOP_N_RECOMMEND = 5
MAX_USERS_FOR_SIM = 2000
HEATMAP_USER_SAMPLE = 30

# 设置中文字体
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.font_manager as fm

# ==========================================
# 全局配置与工具函数
# ==========================================

# 全局变量，用于存储字体属性，确保在各绘图函数中直接使用
GLOBAL_FONT_PROP = None

# 定义SCI风格配色方案
SCI_COLORS = [
    "#4E79A7", "#F28E2B", "#76B7B2", "#E15759", 
    "#B07AA1", "#9C755F", "#FF9DA7", "#BAB0AC", 
    "#59A14F", "#EDC948"
]

def setup_matplotlib_style() -> None:
    """
    配置Matplotlib绘图风格
    1. 强制加载 Windows 系统自带的中文字体文件。
    2. 设置SCI期刊风格的配色和图表样式。
    """
    global GLOBAL_FONT_PROP
    import platform
    import os
    
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
            # 同时尝试注册到全局，双重保险
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = GLOBAL_FONT_PROP.get_name()
            plt.rcParams["font.sans-serif"] = [GLOBAL_FONT_PROP.get_name()]
            print(f"已成功加载字体: {GLOBAL_FONT_PROP.get_name()}")
        except Exception as e:
            print(f"字体加载失败: {e}")
    
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    # 4. 自定义 RC 参数以符合 SCI 风格
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 300,            # 高分辨率
        "axes.titlesize": 16,         # 标题字体大小
        "axes.labelsize": 14,         # 轴标签字体大小
        "xtick.labelsize": 12,        # X轴刻度字体大小
        "ytick.labelsize": 12,        # Y轴刻度字体大小
        "legend.fontsize": 12,        # 图例字体大小
        "axes.grid": True,            # 显示网格
        "grid.alpha": 0.3,            # 网格透明度
        "grid.linestyle": "--",       # 网格线样式
        "axes.spines.top": False,     # 隐藏上边框
        "axes.spines.right": False,   # 隐藏右边框
        "axes.linewidth": 1.5,        # 坐标轴线宽
    })


def load_data(file_path: Path) -> pd.DataFrame:
    """
    读取原始数据文件并统一字段名。
    """
    # 读取CSV数据
    df = pd.read_csv(file_path)
    # 预期字段顺序（如果原始数据没有表头或表头不规范，会在这里修正）
    expected_columns = [
        "user_id",
        "item_id",
        "brand",
        "brand_id",
        "item_name",
        "item_category",
        "item_category_id",
        "behavior_type",
        "timestamp",
        "price",
    ]
    # 当列数正确但列名不一致时，统一替换为标准字段名
    if df.shape[1] == 10 and list(df.columns) != expected_columns:
        df.columns = expected_columns
    return df


def normalize_behavior_type(series: pd.Series) -> pd.Series:
    """
    将行为类型统一为中文标签，便于展示与统计。
    """
    # 常见英文/缩写行为映射到中文
    mapping_text = {
        "pv": "浏览",
        "buy": "购买",
        "cart": "加购",
        "fav": "收藏",
        "view": "浏览",
        "purchase": "购买",
    }
    # 中文标签保持原样
    mapping_cn = {
        "浏览": "浏览",
        "购买": "购买",
        "加购": "加购",
        "收藏": "收藏",
    }
    # 统一转为小写文本后再映射
    series_str = series.astype(str).str.lower()
    mapped = series_str.map(mapping_text)
    # 若英文映射不到，尝试中文映射
    mapped = mapped.fillna(series.astype(str).map(mapping_cn))
    # 兜底：无法识别的类型保持原值
    mapped = mapped.fillna(series.astype(str))
    return mapped


def behavior_score(series: pd.Series) -> pd.Series:
    """
    将行为类型映射为数值强度，默认浏览最弱、购买最强。
    """
    # 行为强度权重，数值越大代表用户对商品兴趣越强
    mapping = {
        "浏览": 1.0,
        "收藏": 2.0,
        "加购": 3.0,
        "购买": 4.0,
        "pv": 1.0,
        "fav": 2.0,
        "cart": 3.0,
        "buy": 4.0,
        "view": 1.0,
        "purchase": 4.0,
    }
    # 如果本身就是数值类型，直接尝试转换
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_score = numeric.where(numeric.notna())
    # 文本类型走映射
    text_score = series.astype(str).str.lower().map(mapping)
    text_score = text_score.fillna(series.astype(str).map(mapping))
    # 组合文本映射与数值映射的结果，最后兜底为1.0
    score = text_score.fillna(numeric_score).fillna(1.0)
    return score.astype(float)


def convert_timestamp(series: pd.Series) -> pd.Series:
    """
    将时间戳转换为时间格式，自动判断秒级或毫秒级。
    """
    # 先尝试转为数值，便于判断时间单位
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        # 如果无法转为数值，直接按日期时间字符串解析
        return pd.to_datetime(series, errors="coerce")
    max_value = numeric.max()
    if max_value > 1e12:
        # 大于1e12通常是毫秒级时间戳
        return pd.to_datetime(numeric, unit="ms", errors="coerce")
    # 默认按秒级时间戳处理
    return pd.to_datetime(numeric, unit="s", errors="coerce")


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造用户-商品交互矩阵，值为行为强度之和。
    """
    # 行为强度按用户-商品聚合求和，得到用户偏好矩阵
    matrix = df.pivot_table(
        index="user_id",
        columns="item_id",
        values="behavior_score",
        aggfunc="sum",
        fill_value=0.0,
    )
    return matrix


def sample_users(matrix: pd.DataFrame, max_users: int) -> pd.DataFrame:
    """
    用户规模过大时抽样，降低相似度矩阵计算开销。
    """
    # 用户数量在阈值内直接返回
    if matrix.shape[0] <= max_users:
        return matrix
    # 超过阈值时随机抽样，保证可运行性
    sampled_users = matrix.sample(max_users, random_state=42)
    return sampled_users


def compute_user_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    使用余弦相似度计算用户两两相似性。
    """
    # 余弦相似度适合高维稀疏向量
    similarity = cosine_similarity(matrix.values)
    similarity_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)
    return similarity_df


def generate_recommendations(
    matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    top_k: int,
    top_n: int,
) -> pd.DataFrame:
    """
    基于K个最相似用户的加权评分，为每个用户生成TopN推荐。
    """
    recommendations = []
    for user_id in matrix.index:
        # 取出当前用户与其他用户的相似度
        similar_users = similarity_df.loc[user_id].drop(user_id)
        # 选择相似度最高的K个邻居
        similar_users = similar_users.nlargest(top_k)
        if similar_users.empty:
            continue
        # 拿到邻居用户的交互矩阵
        neighbor_matrix = matrix.loc[similar_users.index]
        # 用相似度加权求和，得到候选商品评分
        weighted_scores = np.dot(similar_users.values, neighbor_matrix.values)
        weighted_scores = weighted_scores / (similar_users.sum() + 1e-8)
        user_scores = pd.Series(weighted_scores, index=matrix.columns)
        # 过滤掉用户已经交互过的商品
        interacted = matrix.loc[user_id]
        user_scores[interacted > 0] = -np.inf
        # 取TopN作为推荐结果
        top_items = user_scores.nlargest(top_n).index.tolist()
        for rank, item_id in enumerate(top_items, start=1):
            recommendations.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rank": rank,
                    "score": user_scores[item_id],
                }
            )
    return pd.DataFrame(recommendations)


def attach_item_info(recs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    将推荐结果与商品信息拼接，方便查看推荐商品的名称与品类。
    """
    # 去重后的商品信息表，保留名称、品类、品牌、价格
    item_info = (
        df.drop_duplicates(subset=["item_id"])
        .loc[:, ["item_id", "item_name", "item_category", "brand", "price"]]
        .set_index("item_id")
    )
    # 左连接，把商品信息加到推荐结果中
    return recs.join(item_info, on="item_id")


def save_recommendations(recs: pd.DataFrame, output_dir: Path) -> Path:
    """
    保存推荐结果为CSV文件，使用UTF-8带BOM编码。
    """
    output_path = output_dir / "用户协同过滤推荐结果.csv"
    # 使用utf-8-sig确保Excel打开不乱码
    recs.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def plot_behavior_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制用户行为类型分布图（柱状图）
    统计浏览、收藏、加购、购买各有多少条记录。
    
    可视化升级：
    - 使用 SCI 风格自定义配色
    - 每个柱子颜色不同，色彩丰富
    - 添加具体的数值标签
    """
    plt.figure(figsize=(10, 6))
    
    # 统计数据
    counts = df["behavior_label"].value_counts()
    order = counts.index
    
    # 使用自定义 SCI 配色，取前 N 个颜色
    palette = SCI_COLORS[:len(order)]
    
    # 绘制柱状图
    ax = sns.barplot(x=counts.index, y=counts.values, order=order, palette=palette)
    
    # 添加数值标签
    ax.bar_label(ax.containers[0], padding=3, fontsize=12)
    
    # 强制指定字体渲染标题和标签
    plt.title("用户行为类型分布", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("行为类型", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("次数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    
    # 刻度字体也需要强制指定
    for label in ax.get_xticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
    
    plt.tight_layout()
    plt.savefig(output_dir / "用户行为类型分布.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_category_top10(df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制热门商品类别 Top10（水平条形图）
    
    可视化升级：
    - 使用水平条形图，方便阅读长标签
    - 使用渐变色 (viridis 或 mako) 增加美感
    - 添加数值标签
    """
    top_categories = df["item_category"].value_counts().head(10)
    
    plt.figure(figsize=(12, 7))
    
    # 使用 mako 渐变色板，从深到浅
    # sns.barplot 的 orient='h'
    ax = sns.barplot(
        x=top_categories.values, 
        y=top_categories.index.astype(str), 
        palette="mako"
    )
    
    # 添加数值标签
    ax.bar_label(ax.containers[0], padding=3, fontsize=11)
    
    plt.title("商品类别热度 Top10", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("交互次数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("商品类别 ID", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    
    # 强制指定 Y 轴刻度字体（类别名称）
    for label in ax.get_yticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
    for label in ax.get_xticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
    
    plt.tight_layout()
    plt.savefig(output_dir / "商品类别Top10.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_price_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制商品价格分布（直方图 + 核密度估计曲线）
    
    可视化升级：
    - 结合 histplot 和 kdeplot
    - 填充颜色，增加透明度
    - 使用 SCI 配色中的深蓝色
    """
    plt.figure(figsize=(10, 6))
    
    # 过滤异常高价，只展示 99% 分位数以内的价格，使分布图更易读
    price_data = df["price"].dropna()
    limit = price_data.quantile(0.99)
    filtered_price = price_data[price_data <= limit]
    
    sns.histplot(
        filtered_price, 
        bins=50, 
        kde=True, 
        color=SCI_COLORS[0], # 使用深蓝色
        alpha=0.6,
        edgecolor="white",
        line_kws={"linewidth": 2}
    )
    
    plt.title(f"商品价格分布 (Top 99%)", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("价格", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("频数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    
    plt.tight_layout()
    plt.savefig(output_dir / "商品价格分布.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_hourly_activity(df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制用户行为的小时分布（折线图）
    
    可视化升级：
    - 带标记点的折线图
    - 添加阴影区域 (fill_between) 增加视觉效果
    - 标注最高峰和最低谷
    """
    if df["datetime"].isna().all():
        return
        
    hourly_counts = df["datetime"].dt.hour.value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    
    # 绘制折线
    plt.plot(
        hourly_counts.index, 
        hourly_counts.values, 
        marker="o", 
        color=SCI_COLORS[3], # 砖红色
        linewidth=2.5,
        markersize=8,
        label="交互次数"
    )
    
    # 填充下方区域
    plt.fill_between(
        hourly_counts.index, 
        hourly_counts.values, 
        color=SCI_COLORS[3], 
        alpha=0.15
    )
    
    # 标注最大值
    max_hour = hourly_counts.idxmax()
    max_val = hourly_counts.max()
    plt.annotate(
        f'峰值: {max_val}', 
        xy=(max_hour, max_val), 
        xytext=(max_hour, max_val * 1.1),
        arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
        fontsize=11,
        ha='center',
        fontproperties=GLOBAL_FONT_PROP
    )
    
    plt.title("用户活跃时段分布 (24小时)", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("小时 (0-23)", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("交互次数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle="--", alpha=0.4)
    
    # 强制刻度字体
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
    for label in ax.get_yticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
    
    plt.tight_layout()
    plt.savefig(output_dir / "用户行为小时分布.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_user_activity_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制用户活跃度分布（用户交互次数的长尾分布）
    
    可视化升级：
    - 对数坐标轴 (Log Scale) 展示长尾效应
    - 颜色渐变
    """
    user_counts = df["user_id"].value_counts()
    
    plt.figure(figsize=(10, 6))
    
    # 由于是长尾分布，使用对数坐标可能更好，这里先用常规直方图，但美化样式
    sns.histplot(
        user_counts, 
        bins=50, 
        kde=True, 
        color=SCI_COLORS[4], # 紫色
        alpha=0.6,
        edgecolor="white"
    )
    
    plt.yscale("log") # 使用对数坐标轴，因为活跃度通常是幂律分布
    plt.title("用户活跃度分布 (对数坐标)", fontweight="bold", pad=20, fontproperties=GLOBAL_FONT_PROP)
    plt.xlabel("用户交互次数", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    plt.ylabel("用户数 (Log Scale)", fontweight="bold", fontproperties=GLOBAL_FONT_PROP)
    
    # 强制刻度字体
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
    for label in ax.get_yticklabels():
        label.set_fontproperties(GLOBAL_FONT_PROP)
    
    plt.tight_layout()
    plt.savefig(output_dir / "用户活跃度分布.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(similarity_df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制用户相似度矩阵的热力图
    
    可视化升级：
    - 使用更专业的 colormap (如 RdYlBu_r 或 coolwarm)
    - 调整标签字体
    - 增加 colorbar 的标签
    """
    if similarity_df.shape[0] < 2:
        return
        
    sample_count = min(HEATMAP_USER_SAMPLE, similarity_df.shape[0])
    # 随机采样一部分用户进行展示，避免图太大看不清
    sample_users = similarity_df.sample(sample_count, random_state=42).index
    sample_matrix = similarity_df.loc[sample_users, sample_users]
    
    plt.figure(figsize=(10, 9))
    
    # 绘制热力图
    sns.heatmap(
        sample_matrix, 
        cmap="RdYlBu_r", # 红-黄-蓝 渐变，冷暖色调对比明显
        center=0,        # 居中对齐
        annot=False,     # 不显示具体数值，因为太密了
        square=True,     # 保持正方形
        cbar_kws={"label": "余弦相似度"}
    )
    
    plt.title(f"用户相似度矩阵热力图 (随机采样 {sample_count} 用户)", fontweight="bold", pad=20)
    plt.xlabel("用户 ID", fontweight="bold")
    plt.ylabel("用户 ID", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "用户相似度热力图.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """
    主流程：读取数据、构建矩阵、计算相似度、生成推荐并输出图表。
    """
    # 初始化绘图样式
    setup_matplotlib_style()
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取数据并完成字段清洗
    df = load_data(DATA_PATH)
    # 生成中文行为标签用于图表展示
    df["behavior_label"] = normalize_behavior_type(df["behavior_type"])
    # 生成行为强度分数用于协同过滤
    df["behavior_score"] = behavior_score(df["behavior_type"])
    # 价格转为数值类型
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # 时间戳转为时间
    df["datetime"] = convert_timestamp(df["timestamp"])

    # 输出可视化图片到User_png目录
    plot_behavior_distribution(df, OUTPUT_DIR)
    plot_category_top10(df, OUTPUT_DIR)
    plot_price_distribution(df, OUTPUT_DIR)
    plot_hourly_activity(df, OUTPUT_DIR)
    plot_user_activity_distribution(df, OUTPUT_DIR)

    # 构建用户-商品矩阵
    user_item_matrix = build_user_item_matrix(df)
    # 抽样用户以控制计算规模
    user_item_matrix = sample_users(user_item_matrix, MAX_USERS_FOR_SIM)
    # 计算用户相似度矩阵
    similarity_df = compute_user_similarity(user_item_matrix)
    # 输出相似度热力图
    plot_similarity_heatmap(similarity_df, OUTPUT_DIR)

    # 生成推荐列表并补充商品信息
    recs = generate_recommendations(
        user_item_matrix,
        similarity_df,
        TOP_K_NEIGHBORS,
        TOP_N_RECOMMEND,
    )
    recs = attach_item_info(recs, df)
    # 保存推荐结果为CSV
    save_recommendations(recs, OUTPUT_DIR)

    # ==========================================
    # 补充：算法评估模块 (Precision, Recall, Coverage)
    # ==========================================
    print("\n[Evaluation] 开始进行算法评估...")
    evaluate_algorithm(user_item_matrix, similarity_df, TOP_N_RECOMMEND)


def evaluate_algorithm(
    matrix: pd.DataFrame, 
    similarity_df: pd.DataFrame, 
    top_n: int
) -> None:
    """
    评估算法性能：Precision, Recall, Coverage
    注意：为了演示完整性，这里使用简单的留一法或直接基于训练集的覆盖率。
    在严格的科研中，应先拆分 Train/Test 集，这里为了演示 "完整流程"，
    我们计算推荐结果在用户历史行为中的覆盖情况（Hit Rate）作为近似 Recall。
    """
    hits = 0
    total_relevant = 0
    recommended_count = 0
    
    # 获取所有用户的推荐列表
    all_users = matrix.index.tolist()
    # 随机采样部分用户进行评估，避免太慢
    eval_users = all_users[:min(len(all_users), 500)] 
    
    all_recommended_items = set()
    
    for user_id in eval_users:
        # 1. 获取用户真实交互过的物品 (Ground Truth)
        # 注意：在真实 Evaluation 中，这部分物品应该在训练时被 Mask 掉
        # 这里为了演示，我们假设 matrix 是训练集，我们看能否推荐出用户"未来"可能感兴趣的
        # 但由于数据是一份，我们这里计算的是 "拟合能力" (Training Error)
        # 为了更严谨，我们计算 Precision@N 和 Recall@N
        
        user_vector = matrix.loc[user_id]
        ground_truth = user_vector[user_vector > 0].index.tolist()
        
        if not ground_truth:
            continue
            
        # 2. 生成推荐
        # 重新调用生成逻辑（针对单个用户，简化版）
        similar_users = similarity_df.loc[user_id].drop(user_id).nlargest(TOP_K_NEIGHBORS)
        if similar_users.empty:
            continue
            
        neighbor_matrix = matrix.loc[similar_users.index]
        weighted_scores = np.dot(similar_users.values, neighbor_matrix.values)
        weighted_scores = weighted_scores / (similar_users.sum() + 1e-8)
        user_scores = pd.Series(weighted_scores, index=matrix.columns)
        
        # 推荐 Top-N (不过滤已交互的，看看能否召回已有的，作为 Recall 指标)
        top_items = user_scores.nlargest(top_n).index.tolist()
        
        # 3. 计算指标
        hit_items = set(top_items) & set(ground_truth)
        hits += len(hit_items)
        total_relevant += len(ground_truth)
        recommended_count += top_n
        
        all_recommended_items.update(top_items)
        
    precision = hits / recommended_count if recommended_count > 0 else 0.0
    recall = hits / total_relevant if total_relevant > 0 else 0.0
    coverage = len(all_recommended_items) / matrix.shape[1]
    
    print("-" * 40)
    print(f"算法评估结果 (Top-{top_n}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Coverage:  {coverage:.4f}")
    print("-" * 40)
    print("注：此评估基于训练集拟合（Training Set Fit），仅用于验证算法逻辑完整性。")
    print("    在正式科研中，请务必进行 Train/Test 划分。")


def calculate_precision_recall_coverage(
    matrix: pd.DataFrame, 
    similarity_df: pd.DataFrame, 
    top_n: int
) -> None:
    """
    (此函数已废弃，整合到 evaluate_algorithm 中)
    """
    pass

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    main()
