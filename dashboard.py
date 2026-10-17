#!/usr/bin/env python3
"""Serve a small local dashboard for one completed Mininet experiment."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np

from analyze_fairness import summarize as summarize_fairness


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive TCP · Mininet Results</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#121d30;--line:#273650;--text:#e8eef8;--muted:#a9b8cc;--cyan:#45c7d8;--orange:#f2a65a;--green:#76d7a5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}main{max-width:1180px;margin:36px auto;padding:0 22px}h1{margin:0;font-size:clamp(27px,4vw,40px)}h2{font-size:19px;margin:0 0 14px}.sub,.note{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:24px 0}.card,section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}.metric{font-size:26px;font-weight:700;margin-top:5px}.layout{display:grid;grid-template-columns:1.4fr 1fr;gap:14px;margin:14px 0}.chart{width:100%;height:280px}form{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:block;color:var(--muted);font-size:13px}input{width:100%;margin-top:5px;padding:10px;border-radius:8px;border:1px solid var(--line);background:#0b1220;color:var(--text)}button{grid-column:1/-1;padding:11px;border:0;border-radius:8px;background:var(--cyan);color:#06131a;font-weight:700;cursor:pointer}.result{margin-top:16px;padding:14px;border:1px solid var(--line);border-radius:10px;min-height:80px}.result strong{font-size:24px;color:var(--green)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}.pill{display:inline-block;background:#26364d;border-radius:999px;padding:2px 9px;margin:2px;font-size:12px}.foot{font-size:13px;color:var(--muted);margin:22px 0 42px}@media(max-width:760px){.layout{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Adaptive TCP decision engine</h1><div class="sub">Measured Mininet performance · Reno / Cubic / BBR</div>
<div class="grid" id="metrics"></div>
<div class="layout"><section><h2>Measured throughput by bandwidth</h2><canvas id="chart" class="chart"></canvas><div class="note">Mean measured receiver throughput; this summarizes the experiment, not a new network test.</div></section>
<section><h2>Model recommendation</h2><form id="form">
<label>Bandwidth (Mbps)<input name="bandwidth_mbps" type="number" min="1" max="100" step="any" value="20" required></label>
<label>One-way delay (ms)<input name="delay_ms" type="number" min="5" max="200" step="any" value="50" required></label>
<label>Packet loss (%)<input name="loss_pct" type="number" min="0" max="5" step="any" value="1" required></label>
<label>Baseline RTT (ms), optional<input name="baseline_rtt_ms" type="number" min="0.1" step="any" placeholder="estimate from delay"></label>
<button>Get recommendation</button></form><div class="result" id="prediction">Enter conditions to query the trained model.</div>
<div class="note">This is a model prediction from the completed emulated Mininet dataset. It does not change the active TCP algorithm or run live traffic.</div></section></div>
<section><h2>Measured comparison and evaluation</h2><div id="comparison"></div></section>
<section style="margin-top:14px"><h2>Competing flows: queueing, fairness &amp; loaded latency</h2><div class="note" id="fairness-note"></div><div id="fairness-comparison"></div><div id="fairness-paired"></div></section>
<div class="foot">Dataset: <span id="dataset"></span>. Held-out evaluation is split by complete network conditions. Pantheon and synthetic datasets are not mixed into these results.</div>
</main><script>
const fmt=x=>Number(x).toFixed(2);
const ci=x=>x&&x.mean!==null?`${fmt(x.mean)} [${fmt(x.ci95_low)}, ${fmt(x.ci95_high)}]`:'—';
async function load(){const r=await fetch('/api/summary');const d=await r.json();
document.getElementById('dataset').textContent=d.dataset;
document.getElementById('metrics').innerHTML=[['Raw observations',d.rows],['Network conditions',d.conditions],['Runs per algorithm / condition',d.repetitions],['Winning labels',Object.entries(d.wins).map(([k,v])=>`${k.toUpperCase()} ${v}`).join(' · ')]].map(([k,v])=>`<div class="card"><div class="sub">${k}</div><div class="metric">${v}</div></div>`).join('');
const b=d.baselines;document.getElementById('comparison').innerHTML=`<table><thead><tr><th>Strategy / metric</th><th>Held-out mean throughput / score</th></tr></thead><tbody><tr><td>Always Reno</td><td>${fmt(b.always_reno_tp)} Mbps</td></tr><tr><td>Always Cubic</td><td>${fmt(b.always_cubic_tp)} Mbps</td></tr><tr><td>Always BBR</td><td>${fmt(b.always_bbr_tp)} Mbps</td></tr><tr><td>Random Forest adaptive</td><td>${fmt(b.ml_adaptive_tp)} Mbps (${fmt(b.improvement_over_best_fixed_pct)}% vs best fixed)</td></tr><tr><td>Random Forest accuracy</td><td>${fmt(100*d.rf_accuracy)}%</td></tr><tr><td>Decision Tree accuracy</td><td>${fmt(100*d.dt_accuracy)}%</td></tr></tbody></table><p class="note">The held-out set contains ${b.test_conditions} conditions. Results do not show an adaptive throughput gain over always-BBR.</p>`;
renderFairness(d.fairness||[]);
draw(d.bandwidth_means);}
function renderFairness(scenarios){const note=document.getElementById('fairness-note'),by=document.getElementById('fairness-comparison'),paired=document.getElementById('fairness-paired');if(!scenarios.length){note.textContent='No competing-flow CSV loaded. Start dashboard.py with --fairness-csv and the dataset path; repeat the option for additional conditions.';return}note.textContent='Measured Mininet data. Intervals are 95% Student-t confidence intervals over complete runs; results apply only to the displayed conditions.';by.innerHTML=scenarios.map(s=>`<h3>${s.condition} · ${s.dataset}</h3><table><thead><tr><th>Queue</th><th>Competing algorithms</th><th>Total throughput Mbps [95% CI]</th><th>RTT increase ms [95% CI]</th><th>Jain fairness [95% CI]</th><th>Paired runs</th></tr></thead><tbody>${s.groups.map(g=>`<tr><td>${g.queue_discipline}</td><td>${g.variant_pair.toUpperCase()}</td><td>${ci(g.aggregate_throughput_mbps)}</td><td>${ci(g.loaded_rtt_increase_ms)}</td><td>${ci(g.jain_fairness)}</td><td>${g.complete_paired_runs}</td></tr>`).join('')}</tbody></table>`).join('');paired.innerHTML=scenarios.map(s=>`<h3>Paired queue effect · ${s.condition}</h3><p class="note">FQ-CoDel minus pfifo; negative RTT change means lower loaded delay.</p><table><thead><tr><th>Competing algorithms</th><th>Total throughput change Mbps [95% CI]</th><th>RTT increase change ms [95% CI]</th><th>Fairness change [95% CI]</th><th>Paired runs</th></tr></thead><tbody>${s.paired_queue_comparisons.map(g=>`<tr><td>${g.variant_pair.toUpperCase()}</td><td>${ci(g.aggregate_throughput_mbps)}</td><td>${ci(g.loaded_rtt_increase_ms)}</td><td>${ci(g.jain_fairness)}</td><td>${g.paired_repetitions}</td></tr>`).join('')}</tbody></table>`).join('')}
function draw(rows){const c=document.getElementById('chart'),ctx=c.getContext('2d'),ratio=devicePixelRatio||1;c.width=c.clientWidth*ratio;c.height=c.clientHeight*ratio;ctx.scale(ratio,ratio);const w=c.clientWidth,h=c.clientHeight,p={l:46,r:14,t:16,b:36};ctx.clearRect(0,0,w,h);const colors={reno:'#f2a65a',cubic:'#b994f6',bbr:'#45c7d8'},max=Math.max(...rows.flatMap(x=>[x.reno,x.cubic,x.bbr]))*1.12;ctx.font='11px system-ui';ctx.strokeStyle='#273650';ctx.fillStyle='#a9b8cc';for(let i=0;i<=4;i++){let y=p.t+(h-p.t-p.b)*i/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText((max*(1-i/4)).toFixed(0),5,y+4)}const step=(w-p.l-p.r)/rows.length,bw=Math.min(10,step/5);rows.forEach((x,i)=>{let cx=p.l+step*(i+.5);ctx.fillText(x.bandwidth,cx-9,h-12);['reno','cubic','bbr'].forEach((v,j)=>{let bh=(h-p.t-p.b)*x[v]/max;ctx.fillStyle=colors[v];ctx.fillRect(cx+(j-1)*bw*1.2-bw/2,h-p.b-bh,bw,bh)})});let lx=p.l;[['reno',colors.reno],['cubic',colors.cubic],['bbr',colors.bbr]].forEach(([v,col])=>{ctx.fillStyle=col;ctx.fillRect(lx,p.t,9,9);ctx.fillStyle='#a9b8cc';ctx.fillText(v.toUpperCase(),lx+13,p.t+9);lx+=72})}
document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();const f=new FormData(e.target),q=new URLSearchParams();for(const [k,v] of f)if(v!=='')q.set(k,v);const box=document.getElementById('prediction');box.textContent='Predicting…';const r=await fetch('/api/predict?'+q);const d=await r.json();if(!r.ok){box.textContent=d.error;return}const cs=Object.entries(d.confidence).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<span class="pill">${k.toUpperCase()} ${fmt(v*100)}%</span>`).join('');box.innerHTML=`<div>Recommended variant</div><strong>${d.recommended_tcp.toUpperCase()}</strong><div>${cs}</div><div class="note">Baseline RTT used: ${fmt(d.baseline_rtt_ms)} ms</div>`});
load().catch(e=>document.getElementById('metrics').textContent='Could not load experiment summary: '+e);
</script></body></html>"""


def load_fairness_summaries(fairness_paths):
    """Load fairness CSVs separately to preserve each condition's provenance."""
    summaries = []
    for raw_path in fairness_paths:
        path = Path(raw_path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"fairness CSV has no measurements: {path}")
        conditions = sorted({
            (row["bandwidth_mbps"], row["delay_ms"], row["loss_pct"])
            for row in rows
        })
        label = "; ".join(
            f"{bw} Mbps / {delay} ms / {loss}% loss" for bw, delay, loss in conditions
        )
        analysis = summarize_fairness(path)
        if not analysis["groups"]:
            raise ValueError(f"fairness CSV contains no complete paired runs: {path}")
        summaries.append({
            "dataset": path.name,
            "condition": label,
            "groups": analysis["groups"],
            "paired_queue_comparisons": analysis["paired_queue_comparisons"],
        })
    return summaries


def load_summary(raw_path, ml_path, results_path, fairness_paths=None):
    with open(raw_path, newline="") as handle:
        raw = list(csv.DictReader(handle))
    by_bandwidth = defaultdict(lambda: defaultdict(list))
    wins = Counter()
    for row in raw:
        by_bandwidth[float(row["bandwidth_mbps"])][row["tcp_variant"]].append(
            float(row["throughput_mbps"]))
    with open(ml_path, newline="") as handle:
        ml_rows = list(csv.DictReader(handle))
    for row in ml_rows:
        wins[row["best_variant"]] += 1
    with open(results_path) as handle:
        results = json.load(handle)
    counts = Counter((float(row["bandwidth_mbps"]), float(row["delay_ms"]),
                      float(row["loss_pct"]), row["tcp_variant"]) for row in raw)
    reps = sorted(set(counts.values()))
    return {
        "dataset": str(Path(raw_path)), "rows": len(raw),
        "conditions": len(ml_rows), "repetitions": reps[0] if len(reps) == 1 else "unequal",
        "wins": dict(wins),
        "bandwidth_means": [
            {"bandwidth": bw, **{variant: float(np.mean(metrics[variant]))
                                  for variant in ("reno", "cubic", "bbr")}}
            for bw, metrics in sorted(by_bandwidth.items())
        ],
        "baselines": results["baseline_comparison"],
        "rf_accuracy": results["random_forest"]["accuracy"],
        "dt_accuracy": results["decision_tree"]["accuracy"],
        "fairness": load_fairness_summaries(fairness_paths or []),
    }


def make_handler(summary, model):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/api/summary":
                body = json.dumps(summary).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/predict":
                try:
                    params = parse_qs(parsed.query)
                    bw = float(params["bandwidth_mbps"][0])
                    delay = float(params["delay_ms"][0])
                    loss = float(params["loss_pct"][0])
                    rtt = float(params["baseline_rtt_ms"][0]) if params.get("baseline_rtt_ms") else 2 * delay * 1.02
                    if not (1 <= bw <= 100 and 5 <= delay <= 200 and 0 <= loss <= 5 and rtt > 0):
                        raise ValueError("Use bandwidth 1–100 Mbps, delay 5–200 ms, loss 0–5%, and positive RTT.")
                    features = np.array([[bw, delay, loss, rtt]])
                    recommendation = model.predict(features)[0]
                    probabilities = {label: float(score) for label, score in
                                     zip(model.classes_, model.predict_proba(features)[0])}
                    body = json.dumps({"recommended_tcp": recommendation,
                                       "confidence": probabilities,
                                       "baseline_rtt_ms": rtt}).encode()
                    self.send_response(200)
                except (KeyError, ValueError, IndexError) as error:
                    body = json.dumps({"error": str(error)}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
            else:
                body = b"Not found"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string, *args):
            print("[HTTP] " + format_string % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve the Mininet results and TCP predictor dashboard.")
    parser.add_argument("--raw", default="data/dataset_a_mininet_full_recovered.csv")
    parser.add_argument("--ml", default="data/dataset_b_mininet.csv")
    parser.add_argument("--results", default="results/mininet_full/evaluation_results.json")
    parser.add_argument("--model", default="models/mininet_full/random_forest.joblib")
    parser.add_argument("--fairness-csv", action="append", default=[],
                        help="paired-flow Mininet CSV; repeat for additional conditions")
    parser.add_argument("--host", default="127.0.0.1", help="Defaults to localhost for safe local use")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    summary = load_summary(args.raw, args.ml, args.results, args.fairness_csv)
    model = joblib.load(args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(summary, model))
    print(f"Dashboard ready: http://{args.host}:{args.port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
