"""命令行工具：europa-predict 竞彩足球预测解析"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .models import MatchInfo, TeamStat
from .analysis import generate_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SAMPLE = DATA_DIR / "sample_matches.json"


def load_matches(path: str | None = None) -> list[dict]:
    p = Path(path) if path else DEFAULT_SAMPLE
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_match(d: dict) -> MatchInfo:
    return MatchInfo(
        match_id=d.get("match_id", ""),
        home_team=d["home_team"],
        away_team=d["away_team"],
        competition=d.get("competition", "欧罗巴联赛"),
        round_name=d.get("round_name", ""),
        match_time=d.get("match_time", ""),
        handicap=d.get("handicap", 0),
        home_stat=TeamStat(**d["home_stat"]) if d.get("home_stat") else None,
        away_stat=TeamStat(**d["away_stat"]) if d.get("away_stat") else None,
        odds_spf=d.get("odds_spf", {}),
        odds_rqspf=d.get("odds_rqspf", {}),
        odds_zjq=d.get("odds_zjq", {}),
        odds_bqc=d.get("odds_bqc", {}),
    )


def fmt_pct(p: float) -> str:
    return f"{p * 100:5.1f}%"


def print_report(report) -> None:
    m = report.match
    p = report.probs
    print("=" * 70)
    print(f"  {m.home_team}  VS  {m.away_team}")
    print(f"  {m.competition} · {m.round_name} · {m.match_time}")
    print(f"  让球：{m.handicap:+g}  |  比赛编号：{m.match_id}")
    print("=" * 70)

    print("\n【一、胜平负概率 / 赔率对比】")
    print(f"  {'选项':<6}{'模型概率':>10}{'体彩赔率':>10}{'公平赔率':>10}{'隐含概率':>10}")
    fair = report.fair_odds["胜平负"]
    for k, key in [("主胜", "胜"), ("平局", "平"), ("客胜", "负")]:
        prob = getattr(p, {"胜": "home_win_prob", "平": "draw_prob", "负": "away_win_prob"}[key])
        odds = m.odds_spf.get(key, "-")
        imp = (1.0 / odds) if isinstance(odds, (int, float)) and odds else 0
        print(f"  {k:<6}{fmt_pct(prob):>10}{str(odds):>10}{fair[key]:>10.2f}{fmt_pct(imp):>10}")

    print(f"\n  期望进球：主 {p.lambda_home}  客 {p.lambda_away}")
    print(f"  最可能比分：{p.most_likely_score}（概率 {p.most_likely_score_prob * 100:.1f}%）")

    print("\n【二、让球胜平负】")
    fair_rq = report.fair_odds["让球胜平负"]
    for k, key, attr in [("主胜", "胜", "rq_home_win_prob"), ("平局", "平", "rq_draw_prob"), ("客胜", "负", "rq_away_win_prob")]:
        prob = getattr(p, attr)
        odds = m.odds_rqspf.get(key, "-")
        print(f"  {k:<6}{fmt_pct(prob):>10}{str(odds):>10}{fair_rq[key]:>10.2f}")

    print("\n【三、总进球数概率】")
    fair_z = report.fair_odds["总进球"]
    for k in ["0", "1", "2", "3", "4", "5", "6", "7+"]:
        prob = p.total_goals_probs.get(k, 0)
        odds = m.odds_zjq.get(k, "-")
        print(f"  {k:<4}{fmt_pct(prob):>10}{str(odds):>10}{fair_z.get(k, 0):>10.2f}")

    print("\n【四、半全场概率】")
    fair_b = report.fair_odds["半全场"]
    for k, prob in sorted(p.bqc_probs.items(), key=lambda x: x[1], reverse=True):
        odds = m.odds_bqc.get(k, "-")
        print(f"  {k:<6}{fmt_pct(prob):>10}{str(odds):>10}{fair_b.get(k, 0):>10.2f}")

    print("\n【五、价值投注推荐（模型概率 > 市场隐含概率）】")
    if not report.value_bets:
        print("  暂无显著价值投注，建议观望。")
    else:
        print(f"  {'玩法':<12}{'选项':<8}{'模型概率':>10}{'赔率':>8}{'EV':>8}{'信心':>6}{'建议':>10}")
        for vb in report.value_bets[:8]:
            print(f"  {vb.bet_type:<12}{vb.selection:<8}"
                  f"{fmt_pct(vb.model_prob):>10}{vb.odds:>8.2f}"
                  f"{vb.ev:>+8.2%}{vb.confidence:>6}{vb.recommendation:>10}")

    print("\n" + "=" * 70)
    print(f"  摘要：{report.summary}")
    print("=" * 70)


def cmd_list(args) -> None:
    matches = load_matches(args.data)
    print(f"共 {len(matches)} 场比赛：")
    for i, d in enumerate(matches, 1):
        print(f"  [{i}] {d['match_id']}  {d['home_team']} vs {d['away_team']}  "
              f"({d.get('round_name', '')})")


def cmd_predict(args) -> None:
    matches = load_matches(args.data)
    if args.index is not None:
        targets = [matches[args.index - 1]]
    elif args.match_id:
        targets = [d for d in matches if d["match_id"] == args.match_id]
    else:
        targets = matches
    for d in targets:
        report = generate_report(build_match(d))
        print_report(report)


def cmd_predict_json(args) -> None:
    d = json.loads(args.json)
    report = generate_report(build_match(d))
    out = {
        "match_id": report.match.match_id,
        "home": report.match.home_team,
        "away": report.match.away_team,
        "summary": report.summary,
        "spf": {
            "胜": round(report.probs.home_win_prob, 4),
            "平": round(report.probs.draw_prob, 4),
            "负": round(report.probs.away_win_prob, 4),
        },
        "most_likely_score": report.probs.most_likely_score,
        "value_bets": [vars(vb) for vb in report.value_bets],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="europa-predict",
        description="欧罗巴赛事 · 中国体彩竞彩足球职业预测解析系统",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出示例比赛")
    p_list.add_argument("--data", help="自定义比赛数据 JSON 路径")

    p_pred = sub.add_parser("predict", help="输出预测分析报告")
    p_pred.add_argument("--data", help="自定义比赛数据 JSON 路径")
    g = p_pred.add_mutually_exclusive_group()
    g.add_argument("--index", type=int, help="指定比赛序号")
    g.add_argument("--match-id", help="指定比赛 ID")

    p_json = sub.add_parser("predict-json", help="JSON 输入并输出 JSON 结果")
    p_json.add_argument("--json", required=True, help="比赛 JSON 字符串")

    args = parser.parse_args()
    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "predict-json":
        cmd_predict_json(args)


if __name__ == "__main__":
    main()
