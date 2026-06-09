"""
生成 RBAC 权限模型 ER 图 (PNG) — 整洁排版版
2x2 网格布局, 右折线连线, 文本适配框体
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
import os

# ---- 中文字体 ----
_FONT_PATH = None
for _c in ['C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc']:
    if os.path.exists(_c):
        _FONT_PATH = _c
        break

def fp(size=10):
    return FontProperties(fname=_FONT_PATH, size=size) if _FONT_PATH else None

# ========== 配色 ==========
C_BG          = '#F8F9FB'
C_HEADER      = '#1B2A4A'
C_HEADER2     = '#1E40AF'
C_H_TEXT      = '#FFFFFF'
C_BODY_BG     = '#FFFFFF'
C_BORDER      = '#2C3E50'
C_ROW_ALT     = '#F1F3F8'
C_PK          = '#DC2626'
C_FK          = '#2563EB'
C_UQ          = '#D97706'
C_NN          = '#374151'
C_LABEL_BG    = '#FFFFFF'
C_RELATION    = '#059669'

# ========== 实体定义 ==========
TABLES = [
    ("users", "users", C_HEADER2, [
        ("PK",     "id",              "INT AUTO_INCREMENT"),
        ("UQ NN",  "username",        "VARCHAR(80)"),
        ("UQ NN",  "email",           "VARCHAR(120)"),
        ("NN",     "password_hash",   "VARCHAR(256)"),
        ("",       "full_name",       "VARCHAR(120)"),
        ("",       "organization",    "VARCHAR(200)"),
        ("NN",     "role",            "ENUM(admin,researcher,viewer)"),
        ("NN",     "is_active",       "BOOLEAN"),
        ("NN",     "is_verified",     "BOOLEAN"),
        ("UQ",     "api_key",         "VARCHAR(128)"),
        ("",       "last_login_at",   "DATETIME"),
        ("NN",     "created_at",      "DATETIME"),
        ("NN",     "updated_at",      "DATETIME"),
    ]),
    ("datasets", "datasets", C_HEADER, [
        ("PK",     "id",              "INT AUTO_INCREMENT"),
        ("NN",     "name",            "VARCHAR(200)"),
        ("",       "description",     "TEXT"),
        ("NN",     "file_path",       "VARCHAR(512)"),
        ("NN",     "file_format",     "VARCHAR(20)"),
        ("",       "file_size",       "BIGINT"),
        ("NN",     "category",        "VARCHAR(50)"),
        ("",       "row_count",       "INT"),
        ("",       "column_count",    "INT"),
        ("",       "summary_json",    "TEXT"),
        ("NN",     "status",          "ENUM(uploading,ready,processing,error)"),
        ("NN",     "is_public",       "BOOLEAN"),
        ("UQ NN",  "uuid",            "VARCHAR(36)"),
        ("FK NN",  "owner_id",        "INT -> users.id"),
        ("NN",     "created_at",      "DATETIME"),
        ("NN",     "updated_at",      "DATETIME"),
    ]),
    ("model_records", "model_records", C_HEADER, [
        ("PK",     "id",              "INT AUTO_INCREMENT"),
        ("NN",     "name",            "VARCHAR(200)"),
        ("",       "description",     "TEXT"),
        ("",       "version",         "VARCHAR(20)"),
        ("NN",     "model_type",      "ENUM(classification,...)"),
        ("",       "framework",       "VARCHAR(50)"),
        ("",       "model_file_path", "VARCHAR(512)"),
        ("",       "weights_file_path","VARCHAR(512)"),
        ("",       "config_file_path","VARCHAR(512)"),
        ("",       "file_size",       "BIGINT"),
        ("",       "hyperparameters_json","TEXT"),
        ("",       "architecture_json","TEXT"),
        ("",       "metrics_json",    "TEXT"),
        ("",       "accuracy",        "FLOAT"),
        ("",       "precision",       "FLOAT"),
        ("",       "recall",          "FLOAT"),
        ("",       "f1_score",        "FLOAT"),
        ("",       "loss",            "FLOAT"),
        ("",       "r2 / mse / mae",  "FLOAT (regression)"),
        ("FK",     "training_dataset_id","INT -> datasets.id"),
        ("FK",     "training_job_id", "INT -> training_jobs.id"),
        ("NN",     "status",          "ENUM(draft,trained,deployed,archived,failed)"),
        ("NN",     "is_public",       "BOOLEAN"),
        ("",       "deployment_url",  "VARCHAR(512)"),
        ("UQ NN",  "uuid",            "VARCHAR(36)"),
        ("FK NN",  "owner_id",        "INT -> users.id"),
        ("NN",     "created_at",      "DATETIME"),
        ("NN",     "updated_at",      "DATETIME"),
    ]),
    ("training_jobs", "training_jobs", C_HEADER, [
        ("PK",     "id",              "INT AUTO_INCREMENT"),
        ("NN",     "name",            "VARCHAR(200)"),
        ("",       "description",     "TEXT"),
        ("NN",     "task_type",       "ENUM(training,fine_tuning,eval,...)"),
        ("",       "framework",       "VARCHAR(50)"),
        ("NN",     "status",          "ENUM(queued,preparing,running,paused,completed,failed,cancelled)"),
        ("",       "progress_percent","FLOAT"),
        ("",       "current_epoch",   "INT"),
        ("",       "total_epochs",    "INT"),
        ("",       "current_step",    "INT"),
        ("",       "total_steps",     "INT"),
        ("",       "gpu_count",       "INT"),
        ("",       "cpu_cores",       "INT"),
        ("",       "memory_gb",       "FLOAT"),
        ("",       "log_text",        "TEXT"),
        ("",       "error_message",   "TEXT"),
        ("",       "output_dir",      "VARCHAR(512)"),
        ("",       "metrics_history_json","TEXT"),
        ("",       "final_metrics_json","TEXT"),
        ("",       "started_at",      "DATETIME"),
        ("",       "completed_at",    "DATETIME"),
        ("",       "estimated_duration_seconds","INT"),
        ("UQ NN",  "uuid",            "VARCHAR(36)"),
        ("FK NN",  "owner_id",        "INT -> users.id"),
        ("FK",     "dataset_id",      "INT -> datasets.id"),
        ("FK",     "model_id",        "INT -> model_records.id"),
        ("NN",     "created_at",      "DATETIME"),
        ("NN",     "updated_at",      "DATETIME"),
    ]),
]

# ========== 关系: (from, to, from_side, to_side, label) ==========
# 2x2 grid: TL=users, TR=datasets, BL=training_jobs, BR=model_records
RELATIONS = [
    # users -> datasets (TL -> TR, horizontal top)
    ("users",         "datasets",      "right",  "left",   "1 : N", "owner_id"),
    # users -> model_records (TL -> BR, diagonal)
    ("users",         "model_records", "bottom", "top",    "1 : N", "owner_id"),
    # users -> training_jobs (TL -> BL, vertical left)
    ("users",         "training_jobs", "bottom", "top",    "1 : N", "owner_id"),
    # datasets -> training_jobs (TR -> BL, diagonal)
    ("datasets",      "training_jobs", "bottom", "right",  "1 : N", "dataset_id"),
    # datasets -> model_records (TR -> BR, vertical right)
    ("datasets",      "model_records", "bottom", "top",    "1 : N", "training_dataset_id"),
    # training_jobs -> model_records (BL -> BR, horizontal bottom)
    ("training_jobs", "model_records", "right",  "left",   "1 : 1", "model_id"),
    # training_jobs -> model_records (BL -> BR, another)
    ("training_jobs", "model_records", "right",  "left",   "1 : N", "training_job_id"),
]

# ========== 绘制函数 ==========

def draw_table(ax, table_def, x, y, width, row_h=0.185):
    """绘制实体 — 所有文本精确适配在框内"""
    name, title, hdr_color, columns = table_def
    n = len(columns)
    total_h = row_h * n + 0.32  # header + rows

    # 外框阴影
    shadow = FancyBboxPatch(
        (x + 0.06, y - total_h - 0.06), width, total_h,
        boxstyle="round,pad=0.05", linewidth=0, facecolor='#00000012', zorder=1)
    ax.add_patch(shadow)

    # 外框
    box = FancyBboxPatch(
        (x, y - total_h), width, total_h,
        boxstyle="round,pad=0.06", linewidth=2.0,
        edgecolor=C_BORDER, facecolor=C_BODY_BG, zorder=2)
    ax.add_patch(box)

    # 表头
    header_h = 0.35
    header = FancyBboxPatch(
        (x + 0.02, y - header_h + 0.02), width - 0.04, header_h - 0.02,
        boxstyle="round,pad=0.03", linewidth=0, facecolor=hdr_color, zorder=3)
    ax.add_patch(header)
    ax.text(x + width / 2, y - header_h / 2 + 0.01, title,
            ha='center', va='center', fontsize=8, fontweight='bold',
            color=C_H_TEXT, fontproperties=fp(8), zorder=4)

    # 数据行
    for i, (marker, col_name, col_type) in enumerate(columns):
        row_y = y - header_h - row_h * i - row_h / 2

        # 斑马纹
        if i % 2 == 0:
            stripe = plt.Rectangle(
                (x + 0.02, row_y - row_h / 2), width - 0.04, row_h,
                facecolor=C_ROW_ALT, edgecolor='none', zorder=2)
            ax.add_patch(stripe)

        # 标记颜色
        if 'PK' in marker:
            mc, m_bg = C_PK, '#FEE2E2'
        elif 'FK' in marker:
            mc, m_bg = C_FK, '#DBEAFE'
        elif 'UQ' in marker:
            mc, m_bg = C_UQ, '#FEF3C7'
        else:
            mc, m_bg = C_NN, None

        # 标记标签
        if marker:
            tag_w = 0.38 if len(marker) <= 2 else 0.52
            tag = plt.Rectangle(
                (x + 0.06, row_y - 0.065), tag_w, 0.13,
                facecolor=m_bg, edgecolor=mc, linewidth=0.6, zorder=3)
            ax.add_patch(tag)
            ax.text(x + 0.06 + tag_w / 2, row_y, marker,
                    ha='center', va='center', fontsize=5.0, fontweight='bold',
                    color=mc, fontproperties=fp(5), zorder=4)

        # 列名 (固定起始位置)
        cn_x = x + 0.65
        max_name_w = 3.2  # 列名最大宽度
        col_fs = 5.8
        ax.text(cn_x, row_y, col_name, ha='left', va='center',
                fontsize=col_fs, fontweight='bold' if 'FK' in marker else 'normal',
                color='#1F2937', fontproperties=fp(col_fs), zorder=3)

        # 类型 (右对齐, 用更小的字体)
        type_x = x + width - 0.10
        type_fs = 5.2
        # 截断过长类型
        display_type = col_type
        if len(col_type) > 30:
            display_type = col_type[:28] + '..'
        ax.text(type_x, row_y, display_type, ha='right', va='center',
                fontsize=type_fs, color='#6B7280', fontproperties=fp(type_fs),
                zorder=3, style='italic')

    return x, y - total_h, x + width, y


def get_anchor(bounds, side, offset=0):
    """获取锚点，offset 用于同侧多线偏移"""
    x0, y0, x1, y1 = bounds
    base = {
        'left':   (x0, (y0 + y1) / 2),
        'right':  (x1, (y0 + y1) / 2),
        'top':    ((x0 + x1) / 2, y1),
        'bottom': ((x0 + x1) / 2, y0),
    }[side]
    # offset perpendicular to side
    if side in ('left', 'right'):
        return (base[0], base[1] + offset)
    else:
        return (base[0] + offset, base[1])


def draw_relation_ortho(ax, from_b, to_b, from_s, to_s, label, offset_s, offset_t):
    """绘制直角折线关系"""
    p1 = get_anchor(from_b, from_s, offset_s)
    p2 = get_anchor(to_b, to_s, offset_t)

    # 直角折线: 先水平再垂直 (或反过来)
    if from_s in ('left', 'right') and to_s in ('left', 'right'):
        # 水平-水平: 中间点
        mx = (p1[0] + p2[0]) / 2
        path = [(p1[0], p1[1]), (mx, p1[1]), (mx, p2[1]), (p2[0], p2[1])]
    elif from_s in ('top', 'bottom') and to_s in ('top', 'bottom'):
        # 垂直-垂直
        my = (p1[1] + p2[1]) / 2
        path = [(p1[0], p1[1]), (p1[0], my), (p2[0], my), (p2[0], p2[1])]
    elif from_s in ('left', 'right') and to_s in ('top', 'bottom'):
        path = [(p1[0], p1[1]), (p2[0], p1[1]), (p2[0], p2[1])]
    else:
        # top/bottom -> left/right
        path = [(p1[0], p1[1]), (p1[0], p2[1]), (p2[0], p2[1])]

    # 画折线
    xs, ys = zip(*path)
    ax.plot(xs, ys, '-', color=C_RELATION, lw=1.6, zorder=1, alpha=0.85)

    # 标签在路径中点
    mid_idx = len(path) // 2
    lx, ly = path[mid_idx]
    offset_map = {
        ('right','left'): (0.15, 0),
        ('bottom','top'): (0, -0.18),
        ('right','right'): (0.25, 0),
        ('bottom','bottom'): (0, -0.25),
        ('bottom','right'): (0.15, -0.12),
    }
    dx, dy = offset_map.get((from_s, to_s), (0, 0))
    ax.text(lx + dx, ly + dy, label, ha='center', va='center',
            fontsize=6.8, fontweight='bold', color='#065F46',
            fontproperties=fp(6.8),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF',
                      edgecolor='#059669', alpha=0.92, linewidth=0.8),
            zorder=6)


def main():
    # 大画布
    fig, ax = plt.subplots(1, 1, figsize=(44, 28), dpi=150)
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 28)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    # ---- 标题 ----
    fig.suptitle('RBAC Permission Model - Entity Relationship Diagram',
                 fontsize=22, fontweight='bold', y=0.988, color='#111827',
                 fontproperties=fp(22))
    ax.text(22, 27.2, 'Role-Based Access Control  |  2x2 Grid Layout  |  MySQL / SQLAlchemy / Flask-Login',
            ha='center', fontsize=9, color='#6B7280', fontproperties=fp(9), style='italic')

    # ---- 角色卡片 ----
    roles = [
        ("admin",      "Guan Li Yuan", "ALL permissions: CRUD any resource"),
        ("researcher", "Yan Jiu Yuan", "OWN resources: CRUD; VIEW public"),
        ("viewer",     "Guan Cha Zhe",  "READ only: public resources"),
    ]
    card_w, card_h = 3.0, 0.55
    for i, (role, cn, perm) in enumerate(roles):
        cx = 11.5 + i * 3.4
        cy = 26.5
        card = FancyBboxPatch(
            (cx, cy), card_w, card_h,
            boxstyle="round,pad=0.06", linewidth=1.5,
            edgecolor='#7C3AED', facecolor='#F5F3FF', zorder=2)
        ax.add_patch(card)
        ax.text(cx + card_w / 2, cy + card_h - 0.18,
                f"{role} ({cn})", ha='center', va='center',
                fontsize=7, fontweight='bold', color='#7C3AED',
                fontproperties=fp(7), zorder=3)
        ax.text(cx + card_w / 2, cy + 0.12,
                perm, ha='center', va='center', fontsize=5.2,
                color='#6B7280', fontproperties=fp(5.2), zorder=3)

    # ---- 2x2 网格布局 ----
    # TL=users,  TR=datasets
    # BL=training_jobs, BR=model_records
    # margin, gap, table width
    margin = 1.5
    gap_x = 2.5
    gap_y = 3.5
    tbl_w = 10.2

    # 计算各表 y 起始位置 (需要自下而上, 因为表格向下生长)
    # 先算高度
    def calc_h(ncols):
        return 0.185 * ncols + 0.32

    h_users = calc_h(13)          # ~2.725
    h_datasets = calc_h(16)       # ~3.28
    h_models = calc_h(28)         # ~5.50
    h_jobs = calc_h(28)           # ~5.50

    # 顶部行 y 基准
    top_y = 24.5
    # 底部行 y 基准
    bot_y = top_y - max(h_users, h_datasets) - gap_y  # ~24.5 - 3.28 - 3.5 = 17.72

    # x 位置
    x_left = margin
    x_right = margin + tbl_w + gap_x

    positions = {
        "users":         (x_left,  top_y, tbl_w),
        "datasets":      (x_right, top_y, tbl_w),
        "training_jobs": (x_left,  bot_y, tbl_w),
        "model_records": (x_right, bot_y, tbl_w),
    }

    bounds = {}
    for td in TABLES:
        name = td[0]
        x, y, w = positions[name]
        bounds[name] = draw_table(ax, td, x, y, w)

    # ---- 关系连线 (带偏移避免重叠) ----
    # users->datasets: 水平 TL.right -> TR.left
    draw_relation_ortho(ax, bounds["users"], bounds["datasets"],
                        "right", "left", "1 : N", 0, 0)

    # users->training_jobs: 垂直 TL.bottom -> BL.top
    draw_relation_ortho(ax, bounds["users"], bounds["training_jobs"],
                        "bottom", "top", "1 : N", -0.6, -0.6)

    # users->model_records: TL.bottom -> BR.top (需要绕行)
    # 从 users 底部右侧 -> model_records 顶部左侧
    draw_relation_ortho(ax, bounds["users"], bounds["model_records"],
                        "bottom", "top", "1 : N", 0.6, 0.6)

    # datasets->training_jobs: TR.bottom -> BL.right
    draw_relation_ortho(ax, bounds["datasets"], bounds["training_jobs"],
                        "bottom", "right", "1 : N", -0.4, 0)

    # datasets->model_records: TR.bottom -> BR.top
    draw_relation_ortho(ax, bounds["datasets"], bounds["model_records"],
                        "bottom", "top", "1 : N", 0.4, -0.4)

    # training_jobs->model_records: BL.right -> BR.left (两条: 1:1 + 1:N)
    draw_relation_ortho(ax, bounds["training_jobs"], bounds["model_records"],
                        "right", "left", "1 : 1", 0.5, 0.5)

    draw_relation_ortho(ax, bounds["training_jobs"], bounds["model_records"],
                        "right", "left", "1 : N", -0.5, -0.5)

    # ---- 图例 ----
    legend_y = 1.6
    ax.text(1.5, legend_y + 0.4, 'Legend', fontsize=9, fontweight='bold',
            color='#111827', fontproperties=fp(9))
    legend_items = [
        (C_PK, 'PK', 'Primary Key'),
        (C_FK, 'FK', 'Foreign Key'),
        (C_UQ, 'UQ', 'Unique'),
        (C_NN, 'NN', 'Not Null'),
    ]
    for i, (color, abbr, desc) in enumerate(legend_items):
        lx = 1.5 + i * 3.2
        ax.plot(lx, legend_y, 's', color=color, markersize=11,
                markeredgecolor='white', markeredgewidth=1.0)
        ax.text(lx + 0.3, legend_y + 0.08, f'{abbr}  {desc}',
                fontsize=7, fontweight='bold', color=color, fontproperties=fp(7))

    # 关系图例
    ax.plot(14.5, legend_y, 's', color=C_RELATION, markersize=9,
            markeredgecolor='white', markeredgewidth=1.0)
    ax.text(14.8, legend_y - 0.01, 'Relationship (orthogonal routing)',
            fontsize=7, color=C_RELATION, fontproperties=fp(7))

    # 关系说明
    rel_text = (
        "R1 users 1:N datasets (owner_id)  |  "
        "R2 users 1:N model_records (owner_id)  |  "
        "R3 users 1:N training_jobs (owner_id)  |  "
        "R4 datasets 1:N training_jobs (dataset_id)  |  "
        "R5 datasets 1:N model_records (training_dataset_id)  |  "
        "R6 training_jobs 1:1 model_records (model_id)  |  "
        "R7 training_jobs 1:N model_records (training_job_id)"
    )
    ax.text(22, 0.5, rel_text, ha='center', fontsize=6, color='#9CA3AF',
            fontproperties=fp(6))

    ax.text(22, 0.2, 'Generated 2026-06-05  |  myfirstaiproject  |  gitee.com/babat/myfirstaiproject',
            ha='center', fontsize=7, color='#D1D5DB', fontproperties=fp(7))

    # ---- 保存 ----
    output_path = 'er_diagram.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none', pad_inches=0.5)
    plt.close()
    print(f"ER diagram saved: {output_path}")


if __name__ == '__main__':
    main()
