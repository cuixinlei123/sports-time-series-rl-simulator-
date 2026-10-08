"""竞彩赔率解析：反推隐含概率、去除抽水、计算价值投注"""
from __future__ import annotations

from typing import Dict, List

from .models import ValueBet


def implied_probabilities(odds: Dict[str, float]) -> Dict[str, float]:
    """将赔率转化为隐含概率（含抽水）"""
    raw = {k: 1.0 / v for k, v in odds.items() if v and v > 0}
    total = sum(raw.values())
    if total <= 0:
        return {k: 1.0 / len(raw) for k in raw}
    return {k: v / total for k, v in raw.items()}


def bookmaker_margin(odds: Dict[str, float]) -> float:
    """计算庄家抽水率（利润率）"""
    total = sum(1.0 / v for v in odds.values() if v and v > 0)
    return total - 1.0


def fair_odds_from_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """由概率反推公平赔率（无抽水）"""
    return {k: (1.0 / v if v > 0 else 999.0) for k, v in probs.items()}


def detect_value_bets(
    bet_type: str,
    model_probs: Dict[str, float],
    market_odds: Dict[str, float],
    ev_threshold: float = 0.0,
) -> List[ValueBet]:
    """
    检测价值投注。
    EV = model_prob * odds - 1
    若 EV > 0，则模型认为该选项被市场低估。
    """
    results = []
    implied = implied_probabilities(market_odds)
    for selection, m_prob in model_probs.items():
        odds = market_odds.get(selection)
        if odds is None or odds <= 0:
            continue
        ev = m_prob * odds - 1.0
        if ev >= ev_threshold:
            if ev >= 0.15:
                conf = "高"
            elif ev >= 0.05:
                conf = "中"
            else:
                conf = "低"
            if m_prob > 0.5:
                rec = "强烈关注"
            elif m_prob > 0.25:
                rec = "可考虑"
            else:
                rec = "小注试探"
            results.append(ValueBet(
                bet_type=bet_type,
                selection=selection,
                model_prob=round(m_prob, 4),
                implied_prob=round(implied.get(selection, 0), 4),
                odds=odds,
                ev=round(ev, 4),
                confidence=conf,
                recommendation=rec,
            ))
    results.sort(key=lambda x: x.ev, reverse=True)
    return results


def normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        return {k: 1.0 / len(probs) for k in probs}
    return {k: v / total for k, v in probs.items()}
