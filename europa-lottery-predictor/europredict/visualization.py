"""可视化数据生成：为前端 Chart.js 提供图表数据结构"""
from __future__ import annotations

from typing import Dict, List

from .models import AnalysisReport


def spf_chart(report: AnalysisReport) -> dict:
    p = report.probs
    return {
        "type": "doughnut",
        "data": {
            "labels": ["主胜", "平局", "客胜"],
            "datasets": [{
                "data": [round(p.home_win_prob * 100, 1),
                         round(p.draw_prob * 100, 1),
                         round(p.away_win_prob * 100, 1)],
                "backgroundColor": ["#e74c3c", "#f1c40f", "#3498db"],
            }],
        },
    }


def zjq_chart(report: AnalysisReport) -> dict:
    keys = ["0", "1", "2", "3", "4", "5", "6", "7+"]
    return {
        "type": "bar",
        "data": {
            "labels": keys,
            "datasets": [{
                "label": "总进球概率 (%)",
                "data": [round(report.probs.total_goals_probs.get(k, 0) * 100, 1) for k in keys],
                "backgroundColor": "#27ae60",
            }],
        },
    }


def bqc_chart(report: AnalysisReport) -> dict:
    order = ["胜胜", "胜平", "胜负", "平胜", "平平", "平负", "负胜", "负平", "负负"]
    probs = report.probs.bqc_probs
    return {
        "type": "bar",
        "data": {
            "labels": order,
            "datasets": [{
                "label": "半全场概率 (%)",
                "data": [round(probs.get(k, 0) * 100, 2) for k in order],
                "backgroundColor": "#9b59b6",
            }],
        },
    }


def odds_compare_chart(report: AnalysisReport) -> dict:
    """模型概率 vs 市场隐含概率 对比柱状图"""
    p = report.probs
    fair = report.fair_odds["胜平负"]
    market = report.match.odds_spf
    implied = {k: (1.0 / v if v else 0) for k, v in market.items()}
    model = {"胜": p.home_win_prob, "平": p.draw_prob, "负": p.away_win_prob}
    return {
        "type": "bar",
        "data": {
            "labels": ["主胜", "平局", "客胜"],
            "datasets": [
                {"label": "模型概率 (%)", "data": [round(model["胜"]*100,1), round(model["平"]*100,1), round(model["负"]*100,1)], "backgroundColor": "#16a085"},
                {"label": "市场隐含概率 (%)", "data": [round(implied.get("胜",0)*100,1), round(implied.get("平",0)*100,1), round(implied.get("负",0)*100,1)], "backgroundColor": "#c0392b"},
            ],
        },
    }


def build_all_charts(report: AnalysisReport) -> Dict[str, dict]:
    return {
        "spf": spf_chart(report),
        "zjq": zjq_chart(report),
        "bqc": bqc_chart(report),
        "odds_compare": odds_compare_chart(report),
    }
