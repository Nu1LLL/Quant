"""A26：缠中说禅"分型-笔-中枢"结构性alpha。

用户提供了《缠中说禅：教你炒股票108课》原著作为参考。这里只实现
该理论里**客观、可机械判定**的结构层——分型(fractal)、笔(stroke)、
中枢(pivot)，不实现背驰(MACD面积背驰判断)和三类买卖点分类：
后两者需要对"哪个级别的走势做比较"做主观判断，缠中说禅原著里也
承认这需要经验判断，翻译成一套精确的量化规则会引入大量本人的
解读/设计空间，比分型-笔-中枢的机械构造更容易变成"为了让它work
而设计规则"，所以明确不做，如实说明范围。

核心定义（摘自原著，页码/关键词可在提取文本里检索）：

1. K线包含关系处理：相邻两根K线如果一根的高低点完全被另一根包含，
   按当前方向合并——向上时取两根的高点最大值、低点较高者；向下时
   取两根的低点最小值、高点较低者。
2. 顶分型：处理包含关系后，连续三根K线中，中间一根的高点是三者中
   最高、低点也是三者中最高。底分型相反：中间一根低点最低、高点
   也最低。
3. 笔：相邻的顶分型和底分型之间，中间至少间隔一根处理后的K线。
4. 中枢：连续三笔（对应"至少三个连续次级别走势类型"）各自价格
   区间的重叠部分——中枢高(ZG)=三笔区间高点的最小值，
   中枢低(ZD)=三笔区间低点的最大值，只有ZG>ZD时才是有效中枢。

因果性（避免未来数据泄漏）是这里最容易出错、也最重要的一点：
一根K线是否被合并进某个"合并K线"，要等到出现下一根不再合并的
K线才能确定；进一步，一个分型要等它两侧的合并K线都不再变化
（也就是再往后出现了一根新的独立合并K线）才算"确认"。这里在
每一步都记录"这个分型/笔/中枢，最早在原始K线的第几行就已经可以
安全使用"，alpha只使用严格早于当前行的、已确认的结构。
"""
import numpy as np
import pandas as pd

from . import base


def merge_inclusive_bars(df):
    """处理K线包含关系，返回(合并高, 合并低, 每个合并K线起始的原始行号)。

    方向判断：只有当出现两根不构成包含关系的合并K线时才更新方向
    （用高点谁更高判断），这是原著里没有给出完全形式化定义、本模块
    自己补充的操作化规则，采用的是Chan理论量化实现里的通行做法，
    不是为了让某个历史区间"work"而设计的。
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(highs)

    merged_high = []
    merged_low = []
    merged_start_idx = []

    direction = 0

    for i in range(n):
        h, l = highs[i], lows[i]

        if not merged_high:
            merged_high.append(h)
            merged_low.append(l)
            merged_start_idx.append(i)
            continue

        prev_h, prev_l = merged_high[-1], merged_low[-1]
        contains = (h <= prev_h and l >= prev_l) or (h >= prev_h and l <= prev_l)

        if contains:
            if direction >= 0:
                merged_high[-1] = max(h, prev_h)
                merged_low[-1] = max(l, prev_l)
            else:
                merged_high[-1] = min(h, prev_h)
                merged_low[-1] = min(l, prev_l)
        else:
            if h > prev_h:
                direction = 1
            elif h < prev_h:
                direction = -1
            merged_high.append(h)
            merged_low.append(l)
            merged_start_idx.append(i)

    return (
        np.array(merged_high),
        np.array(merged_low),
        np.array(merged_start_idx)
    )


def detect_fractals(merged_high, merged_low, merged_start_idx):
    """返回分型列表：每个元素是(合并K线序号, 类型, 价格, 确认时的
    原始行号)。类型：1=顶分型，-1=底分型。

    确认时的原始行号 = 再往后一个合并K线开始时的原始行号
    （merged_start_idx[j+2]）——因为分型用到j-1,j,j+1三根合并K线，
    j+1的高低点要等j+2这根合并K线开始（也就是不会再被包含进j+1）
    才最终确定，所以要用merged_start_idx[j+2]，而不是j+1自己的
    起始行号。
    """
    m = len(merged_high)
    fractals = []

    for j in range(1, m - 1):
        if j + 2 >= m:
            break
        confirmed_at = int(merged_start_idx[j + 2])

        is_top = (
            merged_high[j] > merged_high[j - 1]
            and merged_high[j] > merged_high[j + 1]
            and merged_low[j] > merged_low[j - 1]
            and merged_low[j] > merged_low[j + 1]
        )
        is_bottom = (
            merged_low[j] < merged_low[j - 1]
            and merged_low[j] < merged_low[j + 1]
            and merged_high[j] < merged_high[j - 1]
            and merged_high[j] < merged_high[j + 1]
        )

        if is_top:
            fractals.append((j, 1, float(merged_high[j]), confirmed_at))
        elif is_bottom:
            fractals.append((j, -1, float(merged_low[j]), confirmed_at))

    return fractals


def build_strokes(fractals, min_merged_gap=1):
    """把分型序列筛成交替出现的顶/底分型（笔的端点）。

    规则：顶和底之间至少间隔min_merged_gap根合并K线（原著里"一定是
    相邻的顶和底"的最低要求）；如果连续出现同类型分型（比如两个顶
    分型之间没有夹着一个底分型），只保留更极端的那个（更高的顶/
    更低的底）——这是原著里没有展开、但几乎所有实操版本都会加的
    标准规则，本模块采用同样的做法，不是自创。
    """
    strokes_points = []

    for merged_idx, kind, price, confirmed_at in fractals:
        if not strokes_points:
            strokes_points.append((merged_idx, kind, price, confirmed_at))
            continue

        last_idx, last_kind, last_price, last_confirmed = strokes_points[-1]

        if kind == last_kind:
            if (kind == 1 and price > last_price) or (
                kind == -1 and price < last_price
            ):
                strokes_points[-1] = (merged_idx, kind, price, confirmed_at)
            continue

        if merged_idx - last_idx > min_merged_gap:
            strokes_points.append((merged_idx, kind, price, confirmed_at))

    return strokes_points


def build_pivots(stroke_points):
    """连续三笔（四个端点）算一个中枢：ZG=三笔区间高点的最小值，
    ZD=三笔区间低点的最大值，只有ZG>ZD才是有效中枢，返回
    (确认时的原始行号, ZG, ZD)的列表。
    """
    pivots = []

    for k in range(len(stroke_points) - 3):
        p0 = stroke_points[k][2]
        p1 = stroke_points[k + 1][2]
        p2 = stroke_points[k + 2][2]
        p3 = stroke_points[k + 3][2]
        confirmed_at = stroke_points[k + 3][3]

        range_highs = [max(p0, p1), max(p1, p2), max(p2, p3)]
        range_lows = [min(p0, p1), min(p1, p2), min(p2, p3)]
        zg = min(range_highs)
        zd = max(range_lows)

        if zg > zd:
            pivots.append((confirmed_at, zg, zd))

    return pivots


def build_pivot_position_alpha(df, scale_multiple=1.0):
    """A26：收盘价相对"最近一个已确认中枢"的位置，连续分数。

    raw = (close - 中枢中点) / (中枢半宽)：0代表在中枢正中间，
    ±1代表刚好在中枢边界，绝对值大于1代表已经脱离中枢区间
    （结构上类似"突破"，但这里只是客观的结构位置，不代表
    缠论里经过背驰确认的第三类买卖点）。

    每一行只使用严格早于该行、已经按前面confirmed_at规则确认过的
    中枢，行之间没有已确认中枢时输出NaN（还没有足够历史形成结构）。
    """
    merged_high, merged_low, merged_start_idx = merge_inclusive_bars(df)
    fractals = detect_fractals(merged_high, merged_low, merged_start_idx)
    stroke_points = build_strokes(fractals)
    pivots = build_pivots(stroke_points)

    n = len(df)
    raw = np.full(n, np.nan)
    close = df["close"].to_numpy()

    pivot_pointer = 0
    current_zg = None
    current_zd = None

    for t in range(n):
        while (
            pivot_pointer < len(pivots)
            and pivots[pivot_pointer][0] <= t
        ):
            current_zg = pivots[pivot_pointer][1]
            current_zd = pivots[pivot_pointer][2]
            pivot_pointer += 1

        if current_zg is None:
            continue

        center = (current_zg + current_zd) / 2.0
        half_width = (current_zg - current_zd) / 2.0
        if half_width > 0:
            raw[t] = (close[t] - center) / half_width

    raw_series = pd.Series(raw, index=df.index)
    normalized = base.squash(raw_series, scale=scale_multiple * 2.0)

    name = "A26_chan_pivot_position"
    return {
        name: base.make_signal(
            name=name,
            raw=raw_series,
            normalized=normalized,
            direction="trend",
            lookback=0,
            literature_reference="缠中说禅《教你炒股票108课》",
            rationale=(
                "收盘价相对最近一个已确认缠论中枢（分型->笔->中枢的"
                "客观结构构造）的位置，只实现理论里可机械判定的结构层，"
                "不含背驰/三类买卖点判断（那部分需要主观级别对比，"
                "刻意不做，避免引入过多本人的解读空间）。"
            ),
            scope="仅分型-笔-中枢，不含背驰/买卖点分类"
        )
    }


def build_alphas(df):
    return build_pivot_position_alpha(df)
