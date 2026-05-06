"""
Generates a self-contained interactive HTML report from a simulation results JSON.

Usage:
    python generate_report.py <results_file.json> [--open]

The report includes:
    - Summary metrics dashboard
    - Interactive network graphs (consensus, ground truth, per-agent mental maps)
    - Affective valence heatmap
    - Per-agent accuracy bar chart
    - Perceived centrality rankings
    - Raw questionnaire excerpts
"""

import argparse
import json
import os
import re
import webbrowser
import html as html_lib
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data helpers (same parsing used in concordia_sim / visualize_css)
# ---------------------------------------------------------------------------


def _parse_json_from_text(text: str):
    """Extract JSON from raw LLM answer text."""
    if not isinstance(text, str):
        return text
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return text


def rebuild_if_needed(data: dict) -> dict:
    """Ensure cognitive_networks and consensus_network are populated."""
    agents = data.get("agents", [])
    cognitive = data.get("cognitive_networks", {})
    consensus = data.get("consensus_network", {})
    raw = data.get("raw_questionnaire_results", [])

    if (not cognitive or not consensus) and raw:
        cognitive = defaultdict(dict)
        for entry in raw:
            agent = entry.get("character")
            dim = entry.get("dimension")
            val = entry.get("value")
            if val is None:
                val = _parse_json_from_text(entry.get("answer_text", ""))
            if agent and dim:
                if dim == "ego_network" and isinstance(val, list):
                    cognitive[agent]["ego"] = val
                elif dim == "global_css" and isinstance(val, dict):
                    cognitive[agent]["global"] = val
                elif dim == "affective_valence" and isinstance(val, dict):
                    cognitive[agent]["valence"] = val
                elif dim == "perceived_centrality" and isinstance(val, list):
                    cognitive[agent]["centrality"] = val
        cognitive = dict(cognitive)
        data["cognitive_networks"] = cognitive

        # Build consensus from global views
        votes = defaultdict(lambda: defaultdict(int))
        for perceiver, d in cognitive.items():
            gv = d.get("global", {})
            if not isinstance(gv, dict):
                continue
            for src, tgts in gv.items():
                if src not in agents or not isinstance(tgts, list):
                    continue
                for tgt in tgts:
                    if tgt in agents:
                        votes[src][tgt] += 1
        data["consensus_network"] = {k: dict(v) for k, v in votes.items()}

    return data


# ---------------------------------------------------------------------------
# Build network edge lists for vis.js
# ---------------------------------------------------------------------------


def consensus_edges(consensus: dict, threshold: int = 0):
    """Return list of {from, to, value} dicts."""
    edges = []
    seen = set()
    for src, targets in consensus.items():
        for tgt, w in targets.items():
            if w <= threshold:
                continue
            pair = tuple(sorted((src, tgt)))
            if pair not in seen:
                seen.add(pair)
                edges.append({"from": src, "to": tgt, "value": w})
    return edges


def ground_truth_edges(gt: dict):
    edges = []
    seen = set()
    for src, targets in gt.items():
        for tgt in targets:
            pair = tuple(sorted((src, tgt)))
            if pair not in seen:
                seen.add(pair)
                edges.append({"from": src, "to": tgt})
    return edges


def agent_global_edges(global_view: dict):
    """Directed edges from one agent's mental map."""
    edges = []
    if not isinstance(global_view, dict):
        return edges
    for src, tgts in global_view.items():
        if isinstance(tgts, list):
            for tgt in tgts:
                edges.append({"from": src, "to": tgt})
    return edges


# ---------------------------------------------------------------------------
# Valence matrix
# ---------------------------------------------------------------------------


def build_valence_matrix(cognitive: dict, agents: list):
    """Return a 2D list [perceiver_idx][target_idx] of valence scores."""
    matrix = []
    for a in agents:
        row = []
        v = cognitive.get(a, {}).get("valence", {})
        for b in agents:
            row.append(v.get(b, None))
        matrix.append(row)
    return matrix


# ---------------------------------------------------------------------------
# Centrality summary
# ---------------------------------------------------------------------------


def centrality_counts(cognitive: dict, agents: list):
    """Count how many times each agent appears in someone's top-3."""
    counts = defaultdict(int)
    for a in agents:
        for name in cognitive.get(a, {}).get("centrality", []):
            counts[name] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(data: dict) -> str:
    agents = data["agents"]
    metrics = data.get("metrics", {})
    cognitive = data.get("cognitive_networks", {})
    consensus = data.get("consensus_network", {})
    gt = data.get("ground_truth_network", {})

    # Prepare JSON payloads for JS
    nodes_json = json.dumps([{"id": a, "label": a} for a in agents])
    consensus_edges_json = json.dumps(consensus_edges(consensus))
    gt_edges_json = json.dumps(ground_truth_edges(gt))

    # Per-agent global views
    agent_global_map = {}
    for a in agents:
        gv = cognitive.get(a, {}).get("global", {})
        agent_global_map[a] = agent_global_edges(gv)
    agent_global_json = json.dumps(agent_global_map)

    # Ego networks
    ego_map = {a: cognitive.get(a, {}).get("ego", []) for a in agents}
    ego_json = json.dumps(ego_map)

    # Valence
    valence_matrix = build_valence_matrix(cognitive, agents)
    valence_json = json.dumps(valence_matrix)

    # Accuracy
    acc = metrics.get("agent_accuracies", {})
    acc_json = json.dumps(acc)

    # Centrality
    cent = centrality_counts(cognitive, agents)
    cent_json = json.dumps(cent)

    # Metrics
    reciprocity = metrics.get("reciprocity", 0)
    transitivity = metrics.get("consensus_transitivity", 0)
    balance = metrics.get("structural_balance", 0)
    cog_acc = metrics.get("town_cognitive_accuracy", 0)

    # Raw questionnaire excerpts
    raw_results = data.get("raw_questionnaire_results", [])
    raw_excerpts = []
    for r in raw_results:
        raw_excerpts.append(
            {
                "character": r.get("character", ""),
                "dimension": r.get("dimension", ""),
                "answer_text": r.get("answer_text", "")[:600],
            }
        )
    raw_json = json.dumps(raw_excerpts)

    setup_name = data.get("setup_name", "Unknown")
    timestamp = data.get("timestamp", "")
    agents_json = json.dumps(agents)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CSS Report — {html_lib.escape(setup_name)} ({timestamp})</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e0e0e6;
    --muted: #8b8fa3;
    --accent: #6c8cff;
    --accent2: #ff6c8c;
    --green: #4caf84;
    --amber: #f0a050;
    --red: #e05050;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; line-height: 1.55; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.25rem; color: var(--accent); margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  h3 {{ font-size: 1rem; color: var(--muted); margin-bottom: 8px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 24px; font-size: 0.95rem; }}
  .grid {{ display: grid; gap: 20px; }}
  .g2 {{ grid-template-columns: 1fr 1fr; }}
  .g3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .g4 {{ grid-template-columns: repeat(4, 1fr); }}
  @media (max-width: 900px) {{ .g2, .g3, .g4 {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
  .metric-card {{ text-align: center; }}
  .metric-val {{ font-size: 2.2rem; font-weight: 700; }}
  .metric-label {{ color: var(--muted); font-size: 0.85rem; margin-top: 4px; }}
  .net-box {{ height: 420px; }}
  select, button {{ background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 14px; font-size: 0.9rem; cursor: pointer; }}
  select:hover, button:hover {{ border-color: var(--accent); }}
  /* Heatmap */
  .heatmap {{ overflow-x: auto; }}
  .heatmap table {{ border-collapse: collapse; font-size: 0.8rem; }}
  .heatmap th, .heatmap td {{ padding: 5px 8px; text-align: center; border: 1px solid var(--border); min-width: 42px; }}
  .heatmap th {{ background: var(--card); color: var(--muted); position: sticky; top: 0; z-index: 2; }}
  .heatmap .row-hdr {{ position: sticky; left: 0; background: var(--card); color: var(--muted); z-index: 1; text-align: right; }}
  /* Bar chart */
  .bar-wrap {{ display: flex; align-items: center; margin-bottom: 6px; font-size: 0.85rem; }}
  .bar-label {{ width: 80px; text-align: right; padding-right: 10px; color: var(--muted); }}
  .bar-track {{ flex: 1; height: 20px; background: var(--border); border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.4s; }}
  .bar-val {{ width: 50px; padding-left: 8px; }}
  /* Excerpts */
  .excerpt {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-bottom: 10px; font-size: 0.82rem; }}
  .excerpt .dim {{ color: var(--accent); font-weight: 600; }}
  .excerpt .agent {{ color: var(--accent2); font-weight: 600; }}
  .excerpt p {{ margin-top: 6px; color: var(--muted); white-space: pre-wrap; }}
  .tabs {{ display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }}
  .tab {{ padding: 5px 12px; border-radius: 6px; font-size: 0.82rem; cursor: pointer; background: var(--bg); border: 1px solid var(--border); }}
  .tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .legend {{ display: flex; gap: 16px; font-size: 0.82rem; color: var(--muted); margin-bottom: 8px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
  .legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<h1>Cognitive Social Structure Report</h1>
<p class="subtitle"><strong>{html_lib.escape(setup_name)}</strong> &mdash; {html_lib.escape(timestamp)}&ensp;|&ensp;{len(agents)} agents</p>

<!-- ==================== METRICS DASHBOARD ==================== -->
<section style="margin-bottom:28px;">
<h2>Summary Metrics</h2>
<div class="grid g4">
  <div class="card metric-card">
    <div class="metric-val" style="color:var(--accent)">{reciprocity:.0%}</div>
    <div class="metric-label">Reciprocity</div>
  </div>
  <div class="card metric-card">
    <div class="metric-val" style="color:var(--green)">{transitivity:.0%}</div>
    <div class="metric-label">Consensus Transitivity</div>
  </div>
  <div class="card metric-card">
    <div class="metric-val" style="color:var(--amber)">{balance:.0%}</div>
    <div class="metric-label">Structural Balance</div>
  </div>
  <div class="card metric-card">
    <div class="metric-val" style="color:var(--accent2)">{cog_acc:.2f}</div>
    <div class="metric-label">Town Cognitive Accuracy</div>
  </div>
</div>
</section>

<!-- ==================== NETWORK GRAPHS ==================== -->
<section style="margin-bottom:28px;">
<h2>Network Graphs</h2>
<div class="grid g2">
  <!-- Consensus -->
  <div class="card">
    <h3>Consensus Network</h3>
    <div class="legend">
      <span>Edge width = vote count</span>
    </div>
    <div id="net-consensus" class="net-box"></div>
  </div>
  <!-- Ground Truth -->
  <div class="card">
    <h3>Ground Truth Network</h3>
    <div class="legend">
      <span>Defined in setup config</span>
    </div>
    <div id="net-truth" class="net-box"></div>
  </div>
</div>

<!-- Agent Mental Map -->
<div class="card" style="margin-top:20px;">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
    <h3 style="margin:0;">Agent Mental Map</h3>
    <select id="agent-select"></select>
  </div>
  <div class="legend">
    <span><span class="dot" style="background:var(--accent)"></span> Global CSS perception</span>
    <span><span class="dot" style="background:var(--accent2)"></span> Ego friends (self-reported)</span>
  </div>
  <div id="net-agent" class="net-box"></div>
</div>
</section>

<!-- ==================== ACCURACY ==================== -->
<section style="margin-bottom:28px;">
<h2>Agent Ego-Network Accuracy (vs Ground Truth)</h2>
<div class="card" id="accuracy-bars"></div>
</section>

<!-- ==================== VALENCE HEATMAP ==================== -->
<section style="margin-bottom:28px;">
<h2>Affective Valence Heatmap</h2>
<div class="card heatmap" id="valence-heatmap"></div>
</section>

<!-- ==================== CENTRALITY ==================== -->
<section style="margin-bottom:28px;">
<h2>Perceived Centrality (Top-3 Mentions)</h2>
<div class="card" id="centrality-bars"></div>
</section>

<!-- ==================== RAW EXCERPTS ==================== -->
<section style="margin-bottom:28px;">
<h2>Questionnaire Excerpts</h2>
<div class="card">
  <div class="tabs" id="excerpt-tabs"></div>
  <div id="excerpt-content" style="max-height:500px;overflow-y:auto;"></div>
</div>
</section>

</div><!-- /container -->

<script>
// ========== DATA ==========
const AGENTS = {agents_json};
const NODES  = {nodes_json};
const CONSENSUS_EDGES = {consensus_edges_json};
const GT_EDGES = {gt_edges_json};
const AGENT_GLOBAL = {agent_global_json};
const EGO_MAP = {ego_json};
const VALENCE = {valence_json};
const ACCURACY = {acc_json};
const CENTRALITY = {cent_json};
const RAW = {raw_json};

// ========== VIS.JS HELPERS ==========
function makeNodes(highlight) {{
  return NODES.map(n => ({{
    id: n.id, label: n.label,
    color: {{
      background: highlight && highlight.includes(n.id) ? '#ff6c8c' : '#6c8cff',
      border: highlight && highlight.includes(n.id) ? '#ff6c8c' : '#4a6adf',
      highlight: {{ background: '#8cabff', border: '#6c8cff' }}
    }},
    font: {{ color: '#e0e0e6', size: 13 }},
    shape: 'dot', size: 18
  }}));
}}

const NET_OPTS = {{
  physics: {{ solver: 'forceAtlas2Based', forceAtlas2Based: {{ gravitationalConstant: -60, springLength: 140 }}, stabilization: {{ iterations: 120 }} }},
  edges: {{ color: {{ color: '#4a5568', highlight: '#6c8cff' }}, smooth: {{ type: 'continuous' }} }},
  interaction: {{ hover: true, tooltipDelay: 100 }},
  layout: {{ randomSeed: 42 }}
}};

// ========== DRAW NETWORKS ==========
// Consensus
new vis.Network(
  document.getElementById('net-consensus'),
  {{
    nodes: new vis.DataSet(makeNodes()),
    edges: new vis.DataSet(CONSENSUS_EDGES.map(e => ({{
      from: e.from, to: e.to,
      width: Math.max(1, e.value * 1.5),
      title: e.from + ' — ' + e.to + '  (votes: ' + e.value + ')',
      color: {{ color: '#6c8cff88' }}
    }})))
  }},
  {{ ...NET_OPTS }}
);

// Ground Truth
new vis.Network(
  document.getElementById('net-truth'),
  {{
    nodes: new vis.DataSet(makeNodes()),
    edges: new vis.DataSet(GT_EDGES.map(e => ({{
      from: e.from, to: e.to, width: 2.5,
      color: {{ color: '#4caf84' }},
      title: e.from + ' ↔ ' + e.to
    }})))
  }},
  {{ ...NET_OPTS }}
);

// Agent selector
const sel = document.getElementById('agent-select');
AGENTS.forEach(a => {{ const o = document.createElement('option'); o.value = a; o.textContent = a; sel.appendChild(o); }});
let agentNet = null;
function drawAgent(name) {{
  const ego = EGO_MAP[name] || [];
  const globalEdges = (AGENT_GLOBAL[name] || []).map(e => ({{
    from: e.from, to: e.to, arrows: 'to',
    color: {{ color: '#6c8cff88' }}, width: 1.5,
    title: e.from + ' → ' + e.to + ' (perceived by ' + name + ')'
  }}));
  // Add ego edges in a different colour
  ego.forEach(f => globalEdges.push({{
    from: name, to: f, width: 3, dashes: [6,3],
    color: {{ color: '#ff6c8c' }},
    title: name + ' considers ' + f + ' a friend (ego)'
  }}));
  const data = {{
    nodes: new vis.DataSet(makeNodes([name])),
    edges: new vis.DataSet(globalEdges)
  }};
  if (agentNet) agentNet.destroy();
  agentNet = new vis.Network(document.getElementById('net-agent'), data, {{ ...NET_OPTS, edges: {{ ...NET_OPTS.edges, arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }} }} }});
}}
sel.addEventListener('change', () => drawAgent(sel.value));
drawAgent(AGENTS[0]);

// ========== ACCURACY BARS ==========
(function() {{
  const container = document.getElementById('accuracy-bars');
  const sorted = AGENTS.slice().sort((a,b) => ((ACCURACY[b]||{{}}).f1||0) - ((ACCURACY[a]||{{}}).f1||0));
  sorted.forEach(a => {{
    const d = ACCURACY[a] || {{}};
    const f1 = d.f1 || 0;
    const p = d.precision || 0;
    const r = d.recall || 0;
    const color = f1 >= 0.7 ? 'var(--green)' : f1 >= 0.3 ? 'var(--amber)' : 'var(--red)';
    container.innerHTML += `
      <div class="bar-wrap">
        <div class="bar-label">${{a}}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${{(f1*100).toFixed(0)}}%;background:${{color}}"></div></div>
        <div class="bar-val">F1 ${{f1.toFixed(2)}}</div>
      </div>`;
  }});
}})();

// ========== VALENCE HEATMAP ==========
(function() {{
  const el = document.getElementById('valence-heatmap');
  let html = '<table><tr><th></th>';
  AGENTS.forEach(a => html += '<th>' + a.slice(0,5) + '</th>');
  html += '</tr>';
  AGENTS.forEach((a, i) => {{
    html += '<tr><td class="row-hdr">' + a + '</td>';
    AGENTS.forEach((b, j) => {{
      const val = VALENCE[i][j];
      let bg = 'transparent';
      if (val !== null && val !== undefined) {{
        if (val > 0) bg = `rgba(76,175,132,${{Math.min(val/5,1)*0.8}})`;
        else if (val < 0) bg = `rgba(224,80,80,${{Math.min(Math.abs(val)/5,1)*0.8}})`;
        else bg = 'rgba(255,255,255,0.05)';
      }}
      const display = val !== null && val !== undefined ? val : '–';
      html += `<td style="background:${{bg}}">${{display}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</table>';
  el.innerHTML += html;
}})();

// ========== CENTRALITY ==========
(function() {{
  const el = document.getElementById('centrality-bars');
  const max = Math.max(...AGENTS.map(a => CENTRALITY[a] || 0), 1);
  const sorted = AGENTS.slice().sort((a,b) => (CENTRALITY[b]||0) - (CENTRALITY[a]||0));
  sorted.forEach(a => {{
    const c = CENTRALITY[a] || 0;
    const pct = (c / max * 100).toFixed(0);
    el.innerHTML += `
      <div class="bar-wrap">
        <div class="bar-label">${{a}}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:var(--accent)"></div></div>
        <div class="bar-val">${{c}} votes</div>
      </div>`;
  }});
}})();

// ========== RAW EXCERPTS ==========
(function() {{
  const tabsEl = document.getElementById('excerpt-tabs');
  const contentEl = document.getElementById('excerpt-content');
  let currentAgent = AGENTS[0];

  function render(agent) {{
    currentAgent = agent;
    tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.agent === agent));
    const items = RAW.filter(r => r.character === agent);
    contentEl.innerHTML = items.map(r =>
      `<div class="excerpt"><span class="agent">${{r.character}}</span> &middot; <span class="dim">${{r.dimension}}</span><p>${{escapeHtml(r.answer_text)}}</p></div>`
    ).join('');
  }}

  AGENTS.forEach(a => {{
    const btn = document.createElement('div');
    btn.className = 'tab' + (a === currentAgent ? ' active' : '');
    btn.textContent = a;
    btn.dataset.agent = a;
    btn.onclick = () => render(a);
    tabsEl.appendChild(btn);
  }});
  render(currentAgent);

  function escapeHtml(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML report from simulation results."
    )
    parser.add_argument(
        "results_file", type=str, help="Path to the _results.json file."
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the report in the default browser."
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: file not found: {args.results_file}")
        return

    with open(args.results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = rebuild_if_needed(data)

    html = generate_html(data)

    out_dir = os.path.join(os.path.dirname(args.results_file), "reports")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.results_file))[0]
    out_path = os.path.join(out_dir, f"{base}_report.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to: {out_path}")
    if args.open:
        webbrowser.open(f"file:///{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
