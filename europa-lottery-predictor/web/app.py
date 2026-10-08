"""Flask 网页演示：输入比赛数据 → 现场运行预测引擎 → 可视化展示"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

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


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        match = build_match(data)
        report = generate_report(match)
        charts = build_all_charts(report)
        payload = {
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
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
