"""核心分析引擎：整合 Elo + Poisson + 赔率，生成完整预测报告"""
from __future__ import annotations

from typing import Dict, List

from .models import (
    AnalysisReport, MatchInfo, ProbabilityResult, TeamStat, ValueBet,
)
from .elo import elo_to_win_probs, form_weight
from .poisson import (
    build_score_matrix, estimate_lambda, half_full_time_probs,
    matrix_to_1x2, matrix_to_handicap_1x2, most_likely_score,
    total_goals_distribution,
)
from .odds import (
    detect_value_bets, fair_odds_from_probs, implied_probabilities,
    normalize_probs,
)


def _ensure_stats(match: MatchInfo) -> tuple[TeamStat, TeamStat]:
    """补全默认球队数据"""
    h = match.home_stat or TeamStat(name=match.home_team)
    a = match.away_stat or TeamStat(name=match.away_team)
    return h, a


def compute_probabilities(match: MatchInfo) -> ProbabilityResult:
    """综合 Elo 与 Poisson 计算所有玩法概率"""
    home, away = _ensure_stats(match)

    # 1) Elo 给出实力差基础胜率
    hw, dw, aw = elo_to_win_probs(home.elo, away.elo, home_boost=65.0)

    # 2) 近期状态修正
    h_form = form_weight(home.recent_wins, home.recent_draws, home.recent_losses)
    a_form = form_weight(away.recent_wins, away.recent_draws, away.recent_losses)
    # 按状态比缩放主胜/客胜
    ratio = h_form / max(a_form, 1e-6)
    hw = hw * ratio
    aw = aw / ratio
    spf = normalize_probs({"胜": hw, "平": dw, "负": aw})

    # 3) Poisson 期望进球
    lambda_home, lambda_away = estimate_lambda(
        home.attack, away.attack, home.defense, away.defense,
        home.home_advantage,
    )
    matrix = build_score_matrix(lambda_home, lambda_away, max_goals=8)

    # 用 Poisson 矩阵重新计算胜平负（更准确），再与 Elo 加权融合
    p_home, p_draw, p_away = matrix_to_1x2(matrix)
    # 融合：Elo 权重 0.4，Poisson 权重 0.6
    blended = {
        "胜": 0.4 * spf["胜"] + 0.6 * p_home,
        "平": 0.4 * spf["平"] + 0.6 * p_draw,
        "负": 0.4 * spf["负"] + 0.6 * p_away,
    }
    blended = normalize_probs(blended)

    # 4) 让球胜平负
    rh, rd, ra = matrix_to_handicap_1x2(matrix, match.handicap)
    rqspf = normalize_probs({"胜": rh, "平": rd, "负": ra})

    # 5) 总进球
    zjq = total_goals_distribution(matrix)

    # 6) 半全场
    bqc = half_full_time_probs(lambda_home, lambda_away)

    # 7) 最可能比分
    best_score, best_prob = most_likely_score(matrix)

    # 比分标签（只保留概率>0.5%的常见比分）
    score_labels = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] >= 0.005:
                score_labels.append(f"{i}:{j}")

    result = ProbabilityResult(
        home_win_prob=blended["胜"],
        draw_prob=blended["平"],
        away_win_prob=blended["负"],
        rq_home_win_prob=rqspf["胜"],
        rq_draw_prob=rqspf["平"],
        rq_away_win_prob=rqspf["负"],
        score_matrix=matrix,
        score_labels=score_labels,
        total_goals_probs=zjq,
        bqc_probs=bqc,
        lambda_home=round(lambda_home, 3),
        lambda_away=round(lambda_away, 3),
        most_likely_score=best_score,
        most_likely_score_prob=round(best_prob, 4),
    )
    return result


def generate_report(match: MatchInfo) -> AnalysisReport:
    """生成完整分析报告（含概率、公平赔率、价值投注）"""
    probs = compute_probabilities(match)

    # 公平赔率
    fair = {
        "胜平负": fair_odds_from_probs({
            "胜": probs.home_win_prob, "平": probs.draw_prob, "负": probs.away_win_prob,
        }),
        "让球胜平负": fair_odds_from_probs({
            "胜": probs.rq_home_win_prob, "平": probs.rq_draw_prob, "负": probs.rq_away_win_prob,
        }),
        "总进球": fair_odds_from_probs(probs.total_goals_probs),
        "半全场": fair_odds_from_probs(probs.bqc_probs),
    }

    # 价值投注检测
    value_bets: List[ValueBet] = []
    if match.odds_spf:
        value_bets += detect_value_bets(
            "胜平负",
            {"胜": probs.home_win_prob, "平": probs.draw_prob, "负": probs.away_win_prob},
            match.odds_spf,
        )
    if match.odds_rqspf:
        value_bets += detect_value_bets(
            "让球胜平负",
            {"胜": probs.rq_home_win_prob, "平": probs.rq_draw_prob, "负": probs.rq_away_win_prob},
            match.odds_rqspf,
        )
    if match.odds_zjq:
        value_bets += detect_value_bets("总进球", probs.total_goals_probs, match.odds_zjq)
    if match.odds_bqc:
        value_bets += detect_value_bets("半全场", probs.bqc_probs, match.odds_bqc)

    # 摘要
    top = sorted(
        [("主胜", probs.home_win_prob), ("平局", probs.draw_prob), ("客胜", probs.away_win_prob)],
        key=lambda x: x[1], reverse=True,
    )
    top_str = "、".join(f"{n}({p*100:.1f}%)" for n, p in top)
    summary = (
        f"{match.home_team} vs {match.away_team}（{match.competition}）"
        f"｜最可能比分 {probs.most_likely_score}（{probs.most_likely_score_prob*100:.1f}%）"
        f"｜胜平负概率 {top_str}"
    )

    return AnalysisReport(
        match=match, probs=probs, value_bets=value_bets,
        fair_odds=fair, summary=summary,
    )
