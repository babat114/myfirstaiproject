"""
============================================
AI Platform - 项目架构图生成器
参考微信架构图风格: 1050x720, 浅色调, 分层设计
============================================
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 9.6), dpi=100)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9.6)
ax.set_aspect('equal')
ax.axis('off')

# ============ 配色方案 (参考微信图片浅色调) ============
C = {
    'bg':           '#F5F5F0',
    'layer_bg':     '#E8E8E0',
    'blue_box':     '#5B8BD0',
    'green_box':    '#6BAF7B',
    'orange_box':   '#E8A44C',
    'purple_box':   '#9B7EC4',
    'red_box':      '#D4787A',
    'teal_box':     '#5BAFA0',
    'blue_text':    '#FFFFFF',
    'dark_text':    '#333333',
    'gray_text':    '#777777',
    'border':       '#CCCCCC',
    'arrow':        '#999999',
    'white_box':    '#FFFFFF',
    'light_blue':   '#D6E4F0',
    'light_green':  '#D9ECD5',
    'light_orange': '#FDE8D0',
    'light_purple': '#E5DCF2',
    'light_red':    '#F5DEDE',
    'light_teal':   '#D0EBE6',
}

def draw_box(ax, x, y, w, h, color, text='', text_color='white', fontsize=9,
             bold=True, radius=0.08, alpha=1.0, edge_color=None, linewidth=1.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=edge_color or color,
        linewidth=linewidth, alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    if text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                color=text_color, fontsize=fontsize, fontweight='bold' if bold else 'normal',
                zorder=4)

def draw_layer_bg(ax, x, y, w, h, color, label='', fontsize=9):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=color, edgecolor=C['border'],
        linewidth=1, alpha=0.6, zorder=1, linestyle='--'
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + 0.15, y + h - 0.22, label, fontsize=fontsize,
                color=C['gray_text'], fontweight='bold', zorder=2, alpha=0.8)

def draw_arrow(ax, x1, y1, x2, y2, color=None, lw=1.5, style='->', zorder=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color or C['arrow'],
                                lw=lw, connectionstyle='arc3,rad=0'), zorder=zorder)

# ============ 背景 ============
ax.add_patch(plt.Rectangle((0, 0), 14, 9.6, facecolor=C['bg'], zorder=0))

# ============ 标题 ============
ax.text(7, 9.15, 'AI Model & Dataset Management Platform', ha='center', va='center',
        fontsize=18, fontweight='bold', color=C['dark_text'], zorder=5)
ax.text(7, 8.85, 'AI Model Training Management Platform -- System Architecture',
        ha='center', va='center', fontsize=9, color=C['gray_text'], zorder=5)

# ============ Layer 1: Presentation ============
draw_layer_bg(ax, 0.2, 7.8, 13.6, 0.85, C['light_blue'],
              'Presentation Layer / Client')
draw_box(ax, 0.8, 7.93, 3.6, 0.52, C['blue_box'], 'Web Browser (Flask Templates)', fontsize=8.5)
draw_box(ax, 5.0, 7.93, 3.6, 0.52, C['teal_box'], 'API Client (RESTful)', fontsize=8.5)
draw_box(ax, 9.2, 7.93, 4.2, 0.52, C['purple_box'], 'Auth: JWT Bearer / API Key', fontsize=8.5)

# ============ Layer 2: Web ============
draw_layer_bg(ax, 0.2, 5.6, 13.6, 2.05, C['light_green'],
              'Web Layer / Flask 3.1')
draw_box(ax, 0.6, 6.35, 2.4, 1.05, C['green_box'],
     'Flask Core\nRoutes + Blueprints\nCSRF + CORS', fontsize=7)
draw_box(ax, 3.3, 6.35, 2.4, 1.05, C['orange_box'],
     'Auth Middleware\nSession + JWT\n+ API Key', fontsize=7)
draw_box(ax, 6.0, 6.35, 2.4, 1.05, C['teal_box'],
     'Jinja2 Templates\nVue 3 CDN\n+ Element Plus', fontsize=7)
draw_box(ax, 8.7, 6.35, 2.4, 1.05, C['purple_box'],
     'Decorators Layer\n@api_login_required\n@rate_limit', fontsize=7)
draw_box(ax, 11.4, 6.35, 2.0, 1.05, C['red_box'],
     'Static Assets\nCSS + JS\n+ Chart.js', fontsize=7)

# ============ Layer 3: Services ============
draw_layer_bg(ax, 0.2, 3.35, 13.6, 2.1, C['light_purple'],
              'Service Layer / Business Logic')

svc_data = [
    (0.5, 4.1, 2.0, 1.05, C['blue_box'], 'AuthService\nAuth + User Mgmt'),
    (2.7, 4.1, 2.0, 1.05, C['green_box'], 'DatasetService\nDataset CRUD'),
    (4.9, 4.1, 2.0, 1.05, C['teal_box'], 'ModelService\nModel Mgmt'),
    (7.1, 4.1, 2.0, 1.05, C['orange_box'], 'TrainingService\nJob Scheduling'),
    (9.3, 4.1, 2.0, 1.05, C['purple_box'], 'InferenceService\nPredict + Eval'),
    (11.5, 4.1, 2.0, 1.05, C['red_box'], 'HPTuningService\nGrid + Random CV'),
]
for sx, sy, sw, sh, sc, st in svc_data:
    draw_box(ax, sx, sy, sw, sh, sc, st, fontsize=6.5)

draw_box(ax, 0.5, 3.5, 13.0, 0.45, '#E0D8EC',
     'DatasetImportService : 15 Public Datasets (sklearn + UCI + Kaggle)',
     text_color=C['dark_text'], fontsize=7, edge_color=C['purple_box'], linewidth=1.2)

# ============ Layer 4: Execution Engine ============
draw_layer_bg(ax, 0.2, 1.1, 13.6, 2.1, C['light_orange'],
              'Execution Engine / Training')

draw_box(ax, 0.5, 1.85, 2.6, 1.05, C['orange_box'],
     'TrainingEngine\nSingleton + Threads\nSSE Real-time', fontsize=7)

trainer_data = [
    (3.4, 1.85, 2.0, 1.05, C['blue_box'], 'sklearn\n8 Algorithms'),
    (5.6, 1.85, 2.0, 1.05, C['red_box'], 'PyTorch\nMLP + GPU'),
    (7.8, 1.85, 2.0, 1.05, C['orange_box'], 'TensorFlow\nKeras API'),
    (10.0, 1.85, 1.8, 1.05, C['green_box'], 'ONNX\nRuntime'),
    (12.0, 1.85, 1.5, 1.05, C['purple_box'], 'Other\n4 Algos'),
]
for sx, sy, sw, sh, sc, st in trainer_data:
    draw_box(ax, sx, sy, sw, sh, sc, st, fontsize=7)

algo = 'RF | LR | SVM | KNN | GBDT | DT | Ridge | KNN-R | MLP | Dense+ReLU+BN+Dropout | AdamW | CosineAnnealing'
ax.text(7.2, 1.55, algo, ha='center', va='center', fontsize=5.5, color=C['gray_text'], zorder=4)

# ============ Layer 5: Data ============
draw_layer_bg(ax, 0.2, 0.15, 6.5, 0.8, C['light_red'],
              'Data Layer / Persistence')
draw_box(ax, 0.6, 0.28, 2.8, 0.52, C['red_box'], 'MySQL 8.0 Database', fontsize=8)
draw_box(ax, 3.7, 0.28, 2.6, 0.52, C['orange_box'], 'File System\nuploads/', fontsize=7.5)

draw_layer_bg(ax, 7.0, 0.15, 6.8, 0.8, C['light_teal'],
              'DevOps / Tooling')
draw_box(ax, 7.3, 0.28, 2.0, 0.52, C['teal_box'], 'pytest\n77 Tests', fontsize=7.5)
draw_box(ax, 9.6, 0.28, 2.0, 0.52, C['purple_box'], 'Scripts\nBatch Train', fontsize=7.5)
draw_box(ax, 11.9, 0.28, 1.7, 0.52, C['blue_box'], 'Git\nVersion Ctrl', fontsize=7.5)

# ============ 连接箭头 ============
for bx in [1.5, 4.5, 7.5, 11.5]:
    draw_arrow(ax, bx, 6.35, bx, 5.5, C['arrow'], lw=1.2)

for bx in [1.5, 6.0, 10.0, 13.0]:
    draw_arrow(ax, bx, 3.5, bx, 3.0, C['arrow'], lw=1.2)

# ============ 图例 ============
legend_y = 0.05
items = [
    (0.3, C['blue_box'], 'Flask/Web'), (3.0, C['green_box'], 'Service'),
    (5.7, C['orange_box'], 'Engine'), (8.4, C['purple_box'], 'Auth/API'),
    (11.1, C['red_box'], 'Storage'),
]
for lx, lc, ll in items:
    ax.add_patch(plt.Rectangle((lx, legend_y), 0.15, 0.12, facecolor=lc, zorder=5))
    ax.text(lx + 0.22, legend_y + 0.06, ll, fontsize=7, color=C['gray_text'], va='center', zorder=5)

# ============ 统计 ============
stats = '113 Models | 5 Frameworks | 15 Datasets | 20+ APIs | 77 Tests | 0 Failures'
ax.text(13.8, 0.05, stats, fontsize=6.5, color=C['gray_text'], ha='right', va='center', zorder=5)

# ============ 保存 ============
output_path = 'C:/Users/86180/Desktop/myfirstaiproject/architecture_diagram.png'
plt.tight_layout(pad=0)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=C['bg'],
            edgecolor='none', pad_inches=0.1)
plt.close()
print(f'Architecture diagram saved: {output_path}')
