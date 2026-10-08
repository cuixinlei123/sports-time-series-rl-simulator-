"""欧罗巴竞彩足球预测解析系统 - 数据模型定义"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class BetType(Enum):
    """竞彩足球五大玩法"""
    SPF = "胜平负"            # 1X2
    RQSPF = "让球胜平负"       # Handicap 1X2
    BF = "比分"               # Correct Score
    ZJQ = "总进球"            # Total Goals
    BQC = "半全场"            # Half-time/Full-time


class Side(Enum):
    HOME = "主"
    DRAW = "平"
    AWAY = "客"


@dataclass
class TeamStat:
    """球队近期状态统计（近 N 场）"""
    name: str
    elo: float = 1500.0                 # Elo 评分
    attack: float = 1.0                 # 进攻系数（相对均值）
    defense: float = 1.0                # 防守系数（越低越稳）
    recent_wins: int = 0
    recent_draws: int = 0
    recent_losses: int = 0
    goals_scored: float = 0.0           # 近 N 场场均进球
    goals_conceded: float = 0.0         # 近 N 场场均失球
    home_advantage: float = 0.25        # 主场加成


@dataclass
class MatchInfo:
    """比赛信息"""
    match_id: str
    home_team: str
    away_team: str
    competition: str = "欧罗巴联赛"
    round_name: str = ""
    match_time: str = ""
    handicap: float = 0.0               # 让球数（主队让球为正）
    home_stat: Optional[TeamStat] = None
    away_stat: Optional[TeamStat] = None
    # 体彩官方赔率（SP）
    odds_spf: Dict[str, float] = field(default_factory=dict)   # {"胜":2.1,"平":3.2,"负":3.0}
    odds_rqspf: Dict[str, float] = field(default_factory=dict) # 让球胜平负赔率
    odds_zjq: Dict[str, float] = field(default_factory=dict)    # 总进球赔率 {"0":8.0,...,"7+":50.0}
    odds_bf: Dict[str, float] = field(default_factory=dict)     # 比分赔率 {"1:0":7.5,...}
    odds_bqc: Dict[str, float] = field(default_factory=dict)    # 半全场赔率 {"胜胜":3.5,...}


@dataclass
class ProbabilityResult:
    """模型输出概率结果"""
    # 胜平负
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    # 让球胜平负
    rq_home_win_prob: float = 0.0
    rq_draw_prob: float = 0.0
    rq_away_win_prob: float = 0.0
    # 比分分布 matrix[i][j] = P(主队i球-客队j球)
    score_matrix: List[List[float]] = field(default_factory=list)
    score_labels: List[str] = field(default_factory=list)
    # 总进球
    total_goals_probs: Dict[str, float] = field(default_factory=dict)
    # 半全场
    bqc_probs: Dict[str, float] = field(default_factory=dict)
    # 期望进球
    lambda_home: float = 0.0
    lambda_away: float = 0.0
    most_likely_score: str = ""
    most_likely_score_prob: float = 0.0


@dataclass
class ValueBet:
    """价值投注建议"""
    bet_type: str
    selection: str
    model_prob: float
    implied_prob: float
    odds: float
    ev: float                      # 期望值（每投注1元的期望收益）
    confidence: str = ""           # 高/中/低
    recommendation: str = ""       # 文字建议


@dataclass
class AnalysisReport:
    """完整分析报告"""
    match: MatchInfo
    probs: ProbabilityResult
    value_bets: List[ValueBet] = field(default_factory=list)
    fair_odds: Dict[str, Dict[str, float]] = field(default_factory=dict)  # 各玩法公平赔率
    summary: str = ""
