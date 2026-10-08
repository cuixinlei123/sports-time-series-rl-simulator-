"""Elo 评分系统：基于球队实力计算预期胜率"""
from __future__ import annotations

import math


def expected_score(rating_a: float, rating_b: float) -> float:
    """计算 A 对 B 的预期得分（胜率）"""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,            # 实际得分：胜=1, 平=0.5, 负=0
    k: float = 30.0,
) -> tuple[float, float]:
    """更新双方 Elo 评分"""
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b


def elo_to_win_probs(rating_home: float, rating_away: float, home_boost: float = 60.0) -> tuple[float, float, float]:
    """
    将 Elo 转化为胜/平/负概率。
    由于 Elo 只给出二分类胜率，这里用修正的正态分布近似三分类。
    home_boost: 主场 Elo 加成
    """
    rh = rating_home + home_boost
    ra = rating_away
    diff = rh - ra
    # 用 logistic 近似主队胜率基础值
    p_home_win = 1.0 / (1.0 + 10.0 ** (-diff / 160.0))
    # 平局概率用一个高斯函数近似（实力越接近平局概率越高）
    p_draw = 0.28 * math.exp(-(diff ** 2) / (2 * 120.0 ** 2))
    p_away_win = 1.0 - p_home_win - p_draw
    # 归一化并保证非负
    total = p_home_win + p_draw + p_away_win
    if total <= 0:
        return 1/3, 1/3, 1/3
    return p_home_win / total, p_draw / total, p_away_win / total


def form_weight(recent_wins: int, recent_draws: int, recent_losses: int) -> float:
    """根据近期状态返回 Elo 修正系数（0.9~1.1）"""
    total = max(recent_wins + recent_draws + recent_losses, 1)
    points = recent_wins * 3 + recent_draws * 1
    form = points / (total * 3)          # 0~1
    return 0.9 + 0.2 * form
