"""Poisson 进球模型：计算比分、总进球、胜平负、半全场概率分布"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


def poisson_pmf(k: int, lam: float) -> float:
    """泊松分布概率质量函数"""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def estimate_lambda(
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_advantage: float,
    league_avg_goals: float = 1.35,
) -> Tuple[float, float]:
    """
    估算双方期望进球数。
    league_avg_goals: 联赛场均进球基准（欧罗巴约 2.7 球/场，单方 1.35）
    """
    lambda_home = league_avg_goals * home_attack * away_defense * (1.0 + home_advantage)
    lambda_away = league_avg_goals * away_attack * home_defense * (1.0 - home_advantage * 0.5)
    return max(lambda_home, 0.1), max(lambda_away, 0.1)


def build_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 8) -> List[List[float]]:
    """构建比分概率矩阵 matrix[i][j] = P(主队进i球, 客队进j球)"""
    matrix = []
    for i in range(max_goals + 1):
        row = []
        for j in range(max_goals + 1):
            row.append(poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away))
        matrix.append(row)
    return matrix


def matrix_to_1x2(matrix: List[List[float]]) -> Tuple[float, float, float]:
    """从比分矩阵求胜/平/负概率"""
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            p = matrix[i][j]
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
    return home_win, draw, away_win


def matrix_to_handicap_1x2(matrix: List[List[float]], handicap: float) -> Tuple[float, float, float]:
    """让球胜平负：在比分上加上让球数后判定"""
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            p = matrix[i][j]
            diff = i - j + handicap
            if diff > 0:
                home_win += p
            elif abs(diff) < 1e-9:
                draw += p
            else:
                away_win += p
    return home_win, draw, away_win


def total_goals_distribution(matrix: List[List[float]]) -> Dict[str, float]:
    """总进球数分布：0~6, 7+"""
    n = len(matrix)
    dist = {str(k): 0.0 for k in range(7)}
    dist["7+"] = 0.0
    for i in range(n):
        for j in range(n):
            total = i + j
            p = matrix[i][j]
            if total >= 7:
                dist["7+"] += p
            else:
                dist[str(total)] += p
    return dist


def half_time_estimate(lambda_home: float, lambda_away: float) -> Tuple[float, float]:
    """估算半场期望进球（通常半场占全场的 45%）"""
    return lambda_home * 0.45, lambda_away * 0.45


def half_full_time_probs(
    lambda_home: float, lambda_away: float, max_goals: int = 5
) -> Dict[str, float]:
    """
    半全场胜平负：9 种组合（胜胜/胜平/胜负/平胜/平平/平负/负胜/负平/负负）
    用半场和全场独立泊松近似。
    """
    h_lambda = lambda_home * 0.45
    a_lambda = lambda_away * 0.45
    ht_matrix = build_score_matrix(h_lambda, a_lambda, max_goals)
    ft_matrix = build_score_matrix(lambda_home, lambda_away, max_goals + 3)

    ht_home, ht_draw, ht_away = matrix_to_1x2(ht_matrix)
    ft_home, ft_draw, ft_away = matrix_to_1x2(ft_matrix)

    labels = [
        ("胜胜", ht_home, ft_home),
        ("胜平", ht_home, ft_draw),
        ("胜负", ht_home, ft_away),
        ("平胜", ht_draw, ft_home),
        ("平平", ht_draw, ft_draw),
        ("平负", ht_draw, ft_away),
        ("负胜", ht_away, ft_home),
        ("负平", ht_away, ft_draw),
        ("负负", ht_away, ft_away),
    ]
    # 独立假设下相乘，再归一化
    raw = {name: ph * pf for name, ph, pf in labels}
    total = sum(raw.values())
    if total <= 0:
        total = 1.0
    return {k: v / total for k, v in raw.items()}


def most_likely_score(matrix: List[List[float]]) -> Tuple[str, float]:
    """返回概率最大的比分及其概率"""
    best_p = 0.0
    best = "0:0"
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > best_p:
                best_p = matrix[i][j]
                best = f"{i}:{j}"
    return best, best_p
