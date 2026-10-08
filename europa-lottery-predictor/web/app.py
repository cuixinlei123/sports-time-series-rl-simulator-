"""Flask 网页演示：输入比赛数据 → 现场运行预测引擎 → 可视化展示"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from europredict.analysis import generate_report
from europredict.cli import build_match, load_matches
from europredict.visualization import build_all_charts

app = Flask(__name__, template_folder=str(ROOT / "web" / "templates"))


@app.route("/")
def index():
    matches = load_matches()
    return render_template("index.html", matches=matches)


@app.route("/api/matches")
def api_matches():
    return jsonify(load_matches())


def _build_payload(report) -> dict:
    """将 AnalysisReport 转为前端可用的 JSON 结构"""
    charts = build_all_charts(report)
    return {
        "match_id": report.match.match_id,
        "home_team": report.match.home_team,
        "away_team": report.match.away_team,
        "competition": report.match.competition,
        "round_name": report.match.round_name,
        "match_time": report.match.match_time,
        "handicap": report.match.handicap,
        "summary": report.summary,
        "spf": {
            "胜": round(report.probs.home_win_prob, 4),
            "平": round(report.probs.draw_prob, 4),
            "负": round(report.probs.away_win_prob, 4),
        },
        "rqspf": {
            "胜": round(report.probs.rq_home_win_prob, 4),
            "平": round(report.probs.rq_draw_prob, 4),
            "负": round(report.probs.rq_away_win_prob, 4),
        },
        "most_likely_score": report.probs.most_likely_score,
        "most_likely_score_prob": round(report.probs.most_likely_score_prob, 4),
        "lambda_home": report.probs.lambda_home,
        "lambda_away": report.probs.lambda_away,
        "total_goals": report.probs.total_goals_probs,
        "bqc": report.probs.bqc_probs,
        "fair_odds": report.fair_odds,
        "value_bets": [vars(vb) for vb in report.value_bets],
        "charts": charts,
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        match = build_match(data)
        report = generate_report(match)
        return jsonify(_build_payload(report))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict-all", methods=["POST"])
def api_predict_all():
    """批量预测全部比赛，返回精简对比数据 + 完整结果列表"""
    data = request.get_json(force=True)
    matches_data = data.get("matches") if data else None
    if not matches_data:
        matches_data = load_matches()
    results = []
    for d in matches_data:
        try:
            match = build_match(d)
            report = generate_report(match)
            payload = _build_payload(report)
            results.append(payload)
        except Exception as e:
            results.append({"error": str(e), "home_team": d.get("home_team", "?"), "away_team": d.get("away_team", "?")})
    # 精简对比表
    summary_table = []
    for r in results:
        if "error" in r:
            continue
        summary_table.append({
            "match": f"{r['home_team']} vs {r['away_team']}",
            "home_win": r["spf"]["胜"],
            "draw": r["spf"]["平"],
            "away_win": r["spf"]["负"],
            "most_likely_score": r["most_likely_score"],
            "score_prob": r["most_likely_score_prob"],
            "lambda_home": r["lambda_home"],
            "lambda_away": r["lambda_away"],
            "top_value": r["value_bets"][0]["selection"] + " " + str(r["value_bets"][0]["odds"]) if r["value_bets"] else "-",
            "top_ev": r["value_bets"][0]["ev"] if r["value_bets"] else 0,
            "value_count": len(r["value_bets"]),
        })
    return jsonify({"summary_table": summary_table, "results": results})


@app.route("/api/export", methods=["POST"])
def api_export():
    """导出预测结果为 JSON 文件下载"""
    import urllib.parse
    data = request.get_json(force=True)
    try:
        match = build_match(data)
        report = generate_report(match)
        payload = _build_payload(report)
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        raw_name = f"predict_{match.home_team}_vs_{match.away_team}.json"
        encoded_name = urllib.parse.quote(raw_name)
        return Response(
            content,
            mimetype="application/json",
            headers={"Content-disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
