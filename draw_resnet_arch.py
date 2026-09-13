# draw_resnet_arch.py
# 运行方式：python draw_resnet_arch.py
# 输出文件：images/resnet34_cifar10_architecture.png
#
# 设计思路：
#   以前向传播（forward pass）为主线，自上而下绘制完整流程图。
#   每个残差块明确画出：主分支（卷积层）+ 旁路（残差连接）→ ⊕ 合并。
#   Projection Shortcut（红线）/ Identity Shortcut（蓝线）均可见。

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

# ── 字体设置 ───────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

# ── 配色 ──────────────────────────────────────────────────────────
CLR = {
    'bg':    '#F8F9FA',
    'input': '#2C3E50',   # 输入/输出
    'stem':  '#1558A0',   # Stem Conv1
    'g2':    '#78281F',   # Conv2_x
    'g3':    '#4A235A',   # Conv3_x
    'g4':    '#7D4400',   # Conv4_x
    'g5':    '#145A32',   # Conv5_x
    'head':  '#0E6655',   # Head
    'conv':  '#C0392B',   # 主分支卷积层
    'bn':    '#1E8449',   # BN/ReLU 层
    'relu':  '#2471A3',   # ReLU
    'add':   '#D68910',   # ⊕ 残差相加
    'proj':  '#C0392B',   # Projection Shortcut（红）
    'idn':   '#1A5276',   # Identity Shortcut（蓝）
    'arr':   '#2C3E50',   # 普通箭头
}

# ── 布局参数 ───────────────────────────────────────────────────────
CX    = 5.8     # 主流程中心 x 坐标
BW    = 7.6     # 主流程框宽度
SC_X  = CX + BW / 2 + 0.75   # 残差旁路路径 x 坐标（右侧）
LBL_X = 0.15    # 左侧 shape badge x 坐标

# ── 图形 ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 40), facecolor=CLR['bg'])   # 加宽以给右侧图例留空间
ax = fig.add_axes([0.03, 0.01, 0.94, 0.98])
ax.set_xlim(0, 18)   # 右侧 12~18 留给图例
ax.set_ylim(-3, 35)
ax.axis('off')
ax.set_facecolor(CLR['bg'])


# ─────────────────────────────────────────────────────────────────
# 基础绘图函数
# ─────────────────────────────────────────────────────────────────

def box(cx, cy, w, h, text, color, fs=10, sub='', tc='white', zorder=3):
    """
    绘制圆角矩形信息块。

    参数:
        cx,cy  : float —— 块中心坐标
        w,h    : float —— 宽度、高度
        text   : str   —— 主标题
        color  : str   —— 背景色
        fs     : float —— 字号
        sub    : str   —— 副标题（shape，显示在下方）
        tc     : str   —— 文字颜色
    """
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.06',
        fc=color, ec='white', lw=2.0, zorder=zorder
    ))
    ty = cy + h * 0.14 if sub else cy  # 有副标题时主标题偏上
    ax.text(cx, ty, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=tc, zorder=zorder + 1, linespacing=1.3)
    if sub:
        ax.text(cx, cy - h * 0.27, sub, ha='center', va='center',
                fontsize=fs - 2, color=tc, style='italic', zorder=zorder + 1)


def varr(y1, y2, x=CX, c=CLR['arr'], lw=2.0):
    """绘制向下垂直箭头（从 y1 指向 y2）。"""
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=5)


def harr(x1, x2, y, c=CLR['arr'], lw=2.0):
    """绘制水平箭头（从 x1 指向 x2）。"""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=5)


def hline(x1, x2, y, c, lw=2.0):
    """绘制水平线段（无箭头）。"""
    ax.plot([x1, x2], [y, y], color=c, lw=lw, zorder=3)


def vline(y1, y2, x, c, lw=2.0):
    """绘制垂直线段（无箭头）。"""
    ax.plot([x, x], [y1, y2], color=c, lw=lw, zorder=3)


def circ(cx, cy, r=0.19, label='⊕', color=CLR['add']):
    """绘制圆形节点（⊕ 残差加法）。"""
    ax.add_patch(plt.Circle((cx, cy), r, fc=color, ec='white', lw=1.5, zorder=5))
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=6)


def sbadge(y, text, x=LBL_X):
    """在左侧绘制黄色 shape 说明标签。"""
    ax.text(x, y, text, ha='left', va='center', fontsize=8.0, color='#3D2B00',
            bbox=dict(boxstyle='round,pad=0.22',
                      fc='#FFF8C5', ec='#C8A900', lw=1.2, zorder=5))


def gadd(y_bot, y_top, color):
    """绘制残差组的彩色背景条带。"""
    ax.add_patch(plt.Rectangle(
        (0.1, y_bot), 13.8, y_top - y_bot,
        fc=color, alpha=0.07, ec=color, lw=1.5, ls='--', zorder=0
    ))


def glabel(y, text, color):
    """绘制残差组标题（带边框）。"""
    ax.text(CX, y, text, ha='center', va='center', fontsize=10.5,
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.35',
                      fc='white', ec=color, lw=1.8, zorder=3))


# ─────────────────────────────────────────────────────────────────
# 核心函数：绘制带残差连接的单个残差块
# ─────────────────────────────────────────────────────────────────

def draw_res_block(y_top, is_proj, group_clr, in_sh, out_sh,
                   conv_stride, blk_name):
    """
    绘制一个完整的残差块（含主分支 + 残差旁路箭头）。

    内部 y 坐标布局（从 y_top 向下递减）：
        +0.00 : 输入分叉点（主分支 + 旁路各走一路）
        -0.22 : 卷积层框顶部
        -1.42 : 卷积层框底部（框高 = 1.20）
        -1.62 : ⊕ 节点中心
        -2.05 : ReLU 框中心
        -2.42 : 输出点（返回值）

    参数:
        y_top       : float —— 块顶部 y 坐标（输入进入位置）
        is_proj     : bool  —— True=Projection Shortcut，False=Identity
        group_clr   : str   —— 所属组的颜色
        in_sh       : str   —— 输入 shape 符号
        out_sh      : str   —— 输出 shape 符号
        conv_stride : int   —— 首层卷积步长（1 或 2）
        blk_name    : str   —— 块标题标注

    返回值:
        float —— 输出点 y 坐标，供下一个元素使用
    """
    CONV_H   = 1.20    # 卷积层框高度
    GAP_TOP  = 0.42   # 输入箭头 → 卷积框顶部的间距（原0.22，增大）
    GAP_ADD  = 0.52   # 卷积框底部 → ⊕ 之间的间距（原0.20，最主要问题）
    GAP_RELU = 0.62   # ⊕ → ReLU 之间的间距（原0.43）
    GAP_OUT  = 0.55   # ReLU → 输出点的间距（原0.37）

    y_in  = y_top
    y_ct  = y_top - GAP_TOP - CONV_H / 2   # 卷积框中心 y
    y_cb  = y_top - GAP_TOP - CONV_H        # 卷积框底部 y
    y_add = y_cb - GAP_ADD                  # ⊕ 节点 y
    y_relu= y_add - GAP_RELU                # ReLU 框 y
    y_out = y_relu - GAP_OUT                # 输出 y

    sc = CLR['proj'] if is_proj else CLR['idn']   # 旁路颜色

    # 块标题
    ax.text(CX, y_in + 0.10, blk_name,
            ha='center', va='bottom', fontsize=8.5,
            fontweight='bold', color=group_clr, zorder=4)

    # 输入 shape badge
    sbadge(y_in, f' {in_sh}')

    # 入口箭头由外部连接箭头提供（终点 = 卷积框顶部），此处不重复绘制

    # ── 主分支：卷积层框 ──────────────────────────────────────────
    conv_label = (f'Conv 3×3  stride={conv_stride}  →  BN + ReLU  →  '
                  f'Conv 3×3  stride=1  →  BN')
    box(CX, y_ct, BW, CONV_H, conv_label, CLR['conv'], fs=9.5)

    # 卷积框 → ⊕
    varr(y_cb, y_add + 0.19)

    # ⊕ 节点
    circ(CX, y_add)

    # ⊕ → ReLU
    varr(y_add - 0.19, y_relu + 0.18)

    # ReLU 框
    box(CX, y_relu, 3.2, 0.35, 'ReLU', CLR['relu'], fs=9.5)

    # 出口箭头由外部连接箭头提供（起点 = ReLU框底部），此处不重复绘制

    # 输出 shape badge（位于 ReLU 框底部，外部连接箭头穿过此处继续向下）
    sbadge(y_out, f' {out_sh}')

    # ── 残差旁路（Shortcut）────────────────────────────────────────
    # 路径：输入分叉 → 向右 → 沿 SC_X 向下 → 向左指向 ⊕
    hline(CX + 0.02, SC_X, y_in - 0.02, sc)          # 水平右引
    vline(y_in - 0.02, y_add, SC_X, sc)               # 垂直向下
    harr(SC_X, CX + 0.19, y_add, sc)                  # 水平左入 ⊕

    # 旁路标注
    sc_text = f'1×1 Conv\nstride={conv_stride}' if is_proj else 'Identity\n直通'
    sc_fc   = '#FDECEA' if is_proj else '#EAF4FB'
    ax.text(SC_X + 0.15, (y_in + y_add) / 2, sc_text,
            ha='left', va='center', fontsize=8.5, color=sc,
            bbox=dict(boxstyle='round,pad=0.25',
                      fc=sc_fc, ec=sc, lw=1.3, zorder=5))

    return y_out


def draw_id_compact(y_top, count, shape, color):
    """
    绘制 N 个 Identity Block 的紧凑占位符（避免重复绘制相同结构）。

    参数:
        y_top  : float —— 占位符顶部 y（输入从上方箭头进入）
        count  : int   —— Identity 块数量
        shape  : str   —— 块间 shape（不变）
        color  : str   —— 所属组颜色

    返回值:
        float —— 输出 y 坐标
    """
    GAP   = 0.40   # 框顶部上方留白（外部连接箭头在此区域通过）
    BOX_H = 0.72   # 占位框高度

    # 入口箭头由外部连接箭头提供（终点 = 框顶部 y_top - GAP），此处不重复绘制

    # 占位框
    cy = y_top - GAP - BOX_H / 2
    box(CX, cy, BW, BOX_H,
        f'× {count}   Identity Block   stride=1   shape 不变：{shape}',
        color, fs=9.5)

    # 旁路（蓝色 U 形）：从主流中心分叉 → 沿 SC_X 绕过框 → 带箭头回到主流中心
    # 表示每个 Identity Block 的残差捷径（输入直接加到输出，shape 不变）
    y_box_top = y_top - GAP            # 框顶部 y（连接箭头终点）
    y_box_bot = y_top - GAP - BOX_H   # 框底部 y（连接箭头起点）
    hline(CX + 0.05, SC_X, y_box_top - 0.02, CLR['idn'], lw=1.5)            # 主流→右侧旁路
    vline(y_box_top - 0.02, y_box_bot + 0.02, SC_X, CLR['idn'], lw=1.5)     # 沿 SC_X 下行
    harr(SC_X, CX + 0.05, y_box_bot + 0.02, CLR['idn'])                     # ← 指回主流（残差相加）
    ax.text(SC_X + 0.15, cy,
            'Identity\nShortcut',   # 残差捷径：输入不经变换直接加到输出
            ha='left', va='center', fontsize=8, color=CLR['idn'],
            bbox=dict(boxstyle='round,pad=0.2',
                      fc='#EAF4FB', ec=CLR['idn'], lw=1.2, zorder=5))

    # 出口箭头由外部连接箭头提供（起点 = 框底部 y_box_bot），此处不重复绘制
    y_out = y_top - GAP - BOX_H - 0.32   # 返回值（外部连接箭头从 y_box_bot 开始，shape badge 在此）
    sbadge(y_out, f' {shape}')
    return y_out


# ═════════════════════════════════════════════════════════════════
# 开始绘图（y 从上到下递减）
# ═════════════════════════════════════════════════════════════════

# ── 标题 ─────────────────────────────────────────────────────────
ax.text(7, 34.5, 'ResNet 网络架构图',
        ha='center', fontsize=22, fontweight='bold', color='#1C2833')
ax.text(7, 33.75,
        '前向传播主线  ·  残差连接（Residual Connection）数据流向可视化',
        ha='center', fontsize=11, color='#666', style='italic')

# ── 输入 ─────────────────────────────────────────────────────────
y = 32.8
box(CX, y, BW, 0.65, 'Input', CLR['input'], fs=11.5,
    sub='(B, Cin, H, W)')
sbadge(y, ' Cin = 3   H = W = 32')
y_in_bot = y - 0.325             # Input 框底部（h=0.65，半高=0.325）

# ── Stem（Conv1 + BN + ReLU）────────────────────────────────────
y -= 1.10                        # Input→Conv1 总偏移（原0.82+0.22=1.04，增大至1.10使箭头更明显）
# y 现在是 Conv1 框中心；先确定位置再画箭头
varr(y_in_bot, y + 0.31)         # ← Input 底 → Conv1 顶（h=0.62，半高=0.31）

box(CX, y, BW, 0.62, 'Conv2d  3×3   stride=1   padding=0',
    CLR['stem'], fs=9.5)
sbadge(y, ' (B, C, H1, W1)')
y_conv_bot = y - 0.31            # Conv1 框底部

y -= 1.00                        # 增大间距使箭头可见（原0.72：箭头仅0.135单位；现1.00：箭头0.415单位）
varr(y_conv_bot, y + 0.275)      # ← Conv1 底 → BN+ReLU 顶（h=0.55，半高=0.275）

box(CX, y, BW, 0.55, 'BatchNorm2d  +  ReLU', CLR['bn'], fs=9.5)
sbadge(y, ' C=16  H1=30')
y_prev = y - 0.275               # ← BN+ReLU 底部（连接箭头起点）
y -= 0.55

# ═════════════════════════════════════════════════════════════════
# Conv2_x  （×3 blocks）
# ═════════════════════════════════════════════════════════════════
y -= 0.45                        # Stem 与 Group2 之间的间隙
G2_TOP = y

glabel(y - 0.05,
       'Conv2_x  ·  ×3 Residual Blocks  ·  C → 4C  ·  stride=1  ·  H1 不变',
       CLR['g2'])
y -= 0.60
varr(y_prev, y - 0.42)           # ← 连接箭头：Stem → G2 Block 0（终点=卷积框顶）

# Block 2.0 Projection（C→4C，stride=1，需 1×1 Conv 对齐通道）
y = draw_res_block(y, True, CLR['g2'],
                   '(B, C, H1, W1)', '(B, 4C, H1, W1)',
                   1, 'Block 2.0  [Projection Shortcut]')
y_prev = y
y -= 0.55
varr(y_prev + 0.375, y - 0.40)   # ← 连接箭头：G2 Block 0 → Identity 紧凑块（ReLU底→框顶）

# Block 2.1、2.2 Identity ×2
y = draw_id_compact(y, 2, '(B, 4C, H1, W1)', CLR['g2'])

G2_BOT = y
gadd(G2_BOT, G2_TOP, CLR['g2'])

y_prev = y
y -= 0.55

# ═════════════════════════════════════════════════════════════════
# Conv3_x  （×4 blocks）
# ═════════════════════════════════════════════════════════════════
G3_TOP = y
glabel(y - 0.05,
       'Conv3_x  ·  ×4 Residual Blocks  ·  4C → 8C  ·  stride=2  ·  H1 → H1/2',
       CLR['g3'])
y -= 0.60
varr(y_prev + 0.32, y - 0.42)    # ← 连接箭头：G2 BOT → G3 Block 0（compact底→卷积框顶）

# Block 3.0 Projection（4C→8C，stride=2，降采样 + 通道翻倍）
y = draw_res_block(y, True, CLR['g3'],
                   '(B, 4C, H1, W1)', '(B, 8C, H1/2, W1/2)',
                   2, 'Block 3.0  [Projection Shortcut]')
y_prev = y
y -= 0.55
varr(y_prev + 0.375, y - 0.40)   # ← 连接箭头：G3 Block 0 → Identity 紧凑块（ReLU底→框顶）

# Block 3.1~3.3 Identity ×3
y = draw_id_compact(y, 3, '(B, 8C, H1/2, W1/2)', CLR['g3'])

G3_BOT = y
gadd(G3_BOT, G3_TOP, CLR['g3'])

y_prev = y
y -= 0.55

# ═════════════════════════════════════════════════════════════════
# Conv4_x  （×6 blocks）
# ═════════════════════════════════════════════════════════════════
G4_TOP = y
glabel(y - 0.05,
       'Conv4_x  ·  ×6 Residual Blocks  ·  8C → 16C  ·  stride=2  ·  H1/2 → H1/4',
       CLR['g4'])
y -= 0.60
varr(y_prev + 0.32, y - 0.42)    # ← 连接箭头：G3 BOT → G4 Block 0（compact底→卷积框顶）

# Block 4.0 Projection（8C→16C，stride=2）
y = draw_res_block(y, True, CLR['g4'],
                   '(B, 8C, H1/2, W1/2)', '(B, 16C, H1/4, W1/4)',
                   2, 'Block 4.0  [Projection Shortcut]')
y_prev = y
y -= 0.55
varr(y_prev + 0.375, y - 0.40)   # ← 连接箭头：G4 Block 0 → Identity 紧凑块（ReLU底→框顶）

# Block 4.1~4.5 Identity ×5
y = draw_id_compact(y, 5, '(B, 16C, H1/4, W1/4)', CLR['g4'])

G4_BOT = y
gadd(G4_BOT, G4_TOP, CLR['g4'])

y_prev = y
y -= 0.55

# ═════════════════════════════════════════════════════════════════
# Conv5_x  （×3 blocks）
# ═════════════════════════════════════════════════════════════════
G5_TOP = y
glabel(y - 0.05,
       'Conv5_x  ·  ×3 Residual Blocks  ·  16C → 32C  ·  stride=2  ·  H1/4 → H1/8',
       CLR['g5'])
y -= 0.60
varr(y_prev + 0.32, y - 0.42)    # ← 连接箭头：G4 BOT → G5 Block 0（compact底→卷积框顶）

# Block 5.0 Projection（16C→32C，stride=2）
y = draw_res_block(y, True, CLR['g5'],
                   '(B, 16C, H1/4, W1/4)', '(B, 32C, H1/8, W1/8)',
                   2, 'Block 5.0  [Projection Shortcut]')
y_prev = y
y -= 0.55
varr(y_prev + 0.375, y - 0.40)   # ← 连接箭头：G5 Block 0 → Identity 紧凑块（ReLU底→框顶）

# Block 5.1~5.2 Identity ×2
y = draw_id_compact(y, 2, '(B, 32C, H1/8, W1/8)', CLR['g5'])

G5_BOT = y
gadd(G5_BOT, G5_TOP, CLR['g5'])

y_prev = y
y -= 0.55
varr(y_prev + 0.32, y + 0.30)    # ← 连接箭头：G5 BOT → Head（compact底→AvgPool框顶）

# ═════════════════════════════════════════════════════════════════
# Head
# ═════════════════════════════════════════════════════════════════
box(CX, y, BW, 0.60,
    'AdaptiveAvgPool2d   (output size = 1×1)',
    CLR['head'], fs=9.5)
sbadge(y, ' (B, 32C, 1, 1)')
# 箭头：AvgPool 底部 → Flatten 顶部（显式计算起终点，确保可见）
y_prev = y - 0.30               # AvgPool 底部（h=0.60，半高=0.30）
y -= 1.05                        # 移至 Flatten 中心（增大间距）
varr(y_prev, y + 0.275)          # AvgPool 底 → Flatten 顶

box(CX, y, BW, 0.55, 'Flatten', CLR['head'], fs=9.5)
sbadge(y, ' (B, 32C)')
y_prev = y - 0.275               # Flatten 底部
y -= 1.00                        # 移至 Linear 中心
varr(y_prev, y + 0.30)           # Flatten 底 → Linear 顶

box(CX, y, BW, 0.60, 'Linear   (32C → K)', CLR['head'], fs=9.5)
sbadge(y, ' (B, K)')
y_prev = y - 0.30                # Linear 底部
y -= 1.05                        # 移至 Output 中心
varr(y_prev, y + 0.325)          # Linear 底 → Output 顶

box(CX, y, BW, 0.65, 'Output Logits', CLR['input'], fs=11, sub='(B, K)')
sbadge(y, ' K = 10')

# ─────────────────────────────────────────────────────────────────
# 图例
# ─────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=CLR['input'], label='输入 / 输出节点'),
    mpatches.Patch(color=CLR['stem'],  label='Stem（Conv1 + BN + ReLU）'),
    mpatches.Patch(color=CLR['conv'],  label='主分支（Conv + BN）'),
    mpatches.Patch(color=CLR['relu'],  label='ReLU 激活'),
    mpatches.Patch(color=CLR['add'],   label='⊕ 残差相加'),
    mpatches.Patch(color=CLR['g2'],    label='Conv2_x（×3，C→4C）'),
    mpatches.Patch(color=CLR['g3'],    label='Conv3_x（×4，4C→8C）'),
    mpatches.Patch(color=CLR['g4'],    label='Conv4_x（×6，8C→16C）'),
    mpatches.Patch(color=CLR['g5'],    label='Conv5_x（×3，16C→32C）'),
    mpatches.Patch(color=CLR['head'],  label='Head（AvgPool + Linear）'),
    mpatches.Patch(color=CLR['proj'],  label='Projection Shortcut（红旁路）'),
    mpatches.Patch(color=CLR['idn'],   label='Identity Shortcut（蓝旁路）'),
]
ax.legend(
    handles=legend_patches,
    loc='upper left',
    bbox_to_anchor=(12.2, 35.0),      # 数据坐标：主内容右侧（x>10.5）、标题区域
    bbox_transform=ax.transData,
    ncol=1, fontsize=8.5, framealpha=0.97,
    title='图例', title_fontsize=10,
    edgecolor='#CCCCCC',
    handlelength=1.0, handletextpad=0.6, borderpad=0.7,
)

# ── 符号说明（底部）───────────────────────────────────────────────
ax.text(
    7, -1.0,
    'C=16  ·  4C=64  ·  8C=128  ·  16C=256  ·  32C=512'
    '  ·  H1=30  ·  H1/2=15  ·  H1/4=8  ·  H1/8=4',
    ha='center', va='center', fontsize=9, color='#555555',
    bbox=dict(boxstyle='round,pad=0.3',
              fc='#FDFEFE', ec='#CCCCCC', lw=1.0)
)

# ─────────────────────────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────────────────────────
os.makedirs('images', exist_ok=True)
SAVE_PATH = 'images/resnet34_cifar10_architecture.png'

plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print(f'saved: {SAVE_PATH}')
