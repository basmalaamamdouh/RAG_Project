import os
import traceback
import re
import json
from groq import Groq
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Rich interactive HTML visualizations for each ML concept.
# Each returns a self-contained HTML string (no external deps except CDN).
# ─────────────────────────────────────────────────────────────────────────────

def _html_knn() -> str:
    return """
<div id="knn-viz" style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">K-Nearest Neighbors — Interactive Demo</h3>
<p style="text-align:center;color:#555;font-size:13px">Click anywhere on the canvas to classify a new point using K=3 neighbors</p>
<div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
  <canvas id="knnCanvas" width="420" height="360"
    style="border:1px solid #ddd;border-radius:8px;cursor:crosshair;background:#fafafa"></canvas>
  <div style="min-width:160px">
    <div style="margin-bottom:12px">
      <label style="font-size:13px;font-weight:600">K value: <span id="kVal">3</span></label><br>
      <input type="range" id="kSlider" min="1" max="9" value="3" style="width:140px">
    </div>
    <div id="knnResult" style="padding:10px;border-radius:8px;background:#f0f0ff;font-size:13px;min-height:60px">
      Click on canvas to classify a point
    </div>
    <div style="margin-top:14px;font-size:12px;color:#555">
      <div><span style="color:#e74c3c">●</span> Class A (Red)</div>
      <div><span style="color:#3498db">●</span> Class B (Blue)</div>
      <div><span style="color:#2ecc71">●</span> Class C (Green)</div>
      <div style="margin-top:6px"><span style="color:#f1c40f;font-size:16px">★</span> Query point</div>
    </div>
  </div>
</div>
<script>
(function(){
  const canvas = document.getElementById('knnCanvas');
  const ctx = canvas.getContext('2d');
  const kSlider = document.getElementById('kSlider');
  const kVal = document.getElementById('kVal');
  const result = document.getElementById('knnResult');

  const classColors = ['#e74c3c','#3498db','#2ecc71'];
  const classNames  = ['Class A','Class B','Class C'];

  // Generate fixed training data
  const rng = (seed) => { let s=seed; return ()=>{ s=(s*9301+49297)%233280; return s/233280; }; };
  const rand = rng(42);
  const points = [];
  const centers = [[120,280],[300,280],[210,100]];
  for(let c=0;c<3;c++){
    for(let i=0;i<25;i++){
      points.push({
        x: centers[c][0] + (rand()-0.5)*120,
        y: centers[c][1] + (rand()-0.5)*120,
        cls: c
      });
    }
  }

  let queryPt = null;

  function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

  function draw(highlight=[]){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    // Grid
    ctx.strokeStyle='#eee'; ctx.lineWidth=1;
    for(let x=0;x<canvas.width;x+=40){ ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke(); }
    for(let y=0;y<canvas.height;y+=40){ ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke(); }
    // Points
    points.forEach(p=>{
      const isHL = highlight.some(h=>h===p);
      ctx.beginPath();
      ctx.arc(p.x,p.y, isHL?9:6, 0, Math.PI*2);
      ctx.fillStyle = classColors[p.cls];
      ctx.globalAlpha = isHL?1:0.75;
      ctx.fill();
      if(isHL){ ctx.strokeStyle='#333';ctx.lineWidth=2;ctx.stroke(); }
      ctx.globalAlpha=1;
    });
    // Lines to neighbors
    if(queryPt && highlight.length){
      highlight.forEach(p=>{
        ctx.beginPath();ctx.moveTo(queryPt.x,queryPt.y);ctx.lineTo(p.x,p.y);
        ctx.strokeStyle='rgba(80,80,80,0.4)';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.stroke();ctx.setLineDash([]);
      });
    }
    // Query point
    if(queryPt){
      ctx.beginPath();ctx.arc(queryPt.x,queryPt.y,10,0,Math.PI*2);
      ctx.fillStyle='#f1c40f';ctx.fill();
      ctx.strokeStyle='#333';ctx.lineWidth=2;ctx.stroke();
      ctx.fillStyle='#333';ctx.font='bold 14px sans-serif';ctx.textAlign='center';
      ctx.fillText('?',queryPt.x,queryPt.y+5);
    }
  }

  function classify(pt, k){
    const sorted = points.slice().sort((a,b)=>dist(pt,a)-dist(pt,b));
    const neighbors = sorted.slice(0,k);
    const votes = [0,0,0];
    neighbors.forEach(n=>votes[n.cls]++);
    const winner = votes.indexOf(Math.max(...votes));
    return {winner, neighbors, votes};
  }

  canvas.addEventListener('click', e=>{
    const rect = canvas.getBoundingClientRect();
    queryPt = {x: e.clientX-rect.left, y: e.clientY-rect.top};
    const k = parseInt(kSlider.value);
    const {winner, neighbors, votes} = classify(queryPt, k);
    draw(neighbors);
    result.innerHTML = `
      <b>Prediction:</b> <span style="color:${classColors[winner]};font-weight:700">${classNames[winner]}</span><br>
      <b>Votes (K=${k}):</b><br>
      ${classNames.map((n,i)=>`<span style="color:${classColors[i]}">● ${n}: ${votes[i]}</span>`).join('<br>')}
    `;
  });

  kSlider.addEventListener('input',()=>{
    kVal.textContent = kSlider.value;
    if(queryPt){ canvas.dispatchEvent(new MouseEvent('click',{clientX:queryPt.x+canvas.getBoundingClientRect().left, clientY:queryPt.y+canvas.getBoundingClientRect().top})); }
  });

  draw();
})();
</script>
</div>"""


def _html_svm() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Support Vector Machine — Decision Boundary</h3>
<p style="text-align:center;color:#555;font-size:13px">Drag the margin slider to see how SVM maximizes the margin between classes</p>
<canvas id="svmCanvas" width="500" height="380" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="text-align:center;margin-top:10px">
  <label style="font-size:13px;font-weight:600">Margin width: <span id="mVal">1.5</span></label><br>
  <input type="range" id="mSlider" min="0.5" max="3.0" step="0.1" value="1.5" style="width:200px">
</div>
<script>
(function(){
  const canvas = document.getElementById('svmCanvas');
  const ctx = canvas.getContext('2d');
  const slider = document.getElementById('mSlider');
  const mVal = document.getElementById('mVal');

  // Fixed points: class -1 bottom-left, class +1 top-right
  const pts1 = [{x:80,y:300},{x:120,y:260},{x:60,y:240},{x:150,y:310},{x:100,y:340},{x:140,y:280},{x:70,y:290},{x:180,y:330}];
  const pts2 = [{x:320,y:80},{x:360,y:50},{x:400,y:100},{x:300,y:120},{x:430,y:60},{x:340,y:130},{x:390,y:40},{x:460,y:90}];

  function draw(margin){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    // Background shading
    const grd1 = ctx.createLinearGradient(0,0,canvas.width,canvas.height);
    grd1.addColorStop(0,'rgba(52,152,219,0.08)');
    grd1.addColorStop(0.5,'rgba(255,255,255,0)');
    grd1.addColorStop(1,'rgba(231,76,60,0.08)');
    ctx.fillStyle=grd1; ctx.fillRect(0,0,canvas.width,canvas.height);

    // Decision boundary (diagonal)
    const mx = margin * 40;
    ctx.strokeStyle='#4f46e5'; ctx.lineWidth=2.5; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(50, canvas.height-50); ctx.lineTo(canvas.width-50, 50); ctx.stroke();

    // Margin lines
    ctx.strokeStyle='rgba(79,70,229,0.4)'; ctx.lineWidth=1.5; ctx.setLineDash([6,4]);
    ctx.beginPath(); ctx.moveTo(50-mx, canvas.height-50); ctx.lineTo(canvas.width-50-mx, 50); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(50+mx, canvas.height-50); ctx.lineTo(canvas.width-50+mx, 50); ctx.stroke();
    ctx.setLineDash([]);

    // Margin bracket annotation
    ctx.fillStyle='#4f46e5'; ctx.font='bold 12px sans-serif';
    ctx.fillText(`Margin = ${margin.toFixed(1)}`, canvas.width/2-35, canvas.height/2+20);

    // Points
    pts1.forEach(p=>{
      ctx.beginPath(); ctx.arc(p.x,p.y,7,0,Math.PI*2);
      ctx.fillStyle='rgba(52,152,219,0.85)'; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
    });
    pts2.forEach(p=>{
      ctx.beginPath(); ctx.arc(p.x,p.y,7,0,Math.PI*2);
      ctx.fillStyle='rgba(231,76,60,0.85)'; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
    });

    // Support vectors (points closest to margin)
    const sv1 = pts1[1]; const sv2 = pts2[0];
    [sv1,sv2].forEach(p=>{
      ctx.beginPath(); ctx.arc(p.x,p.y,11,0,Math.PI*2);
      ctx.strokeStyle='#f39c12'; ctx.lineWidth=2.5; ctx.stroke();
    });

    // Legend
    ctx.font='12px sans-serif'; ctx.fillStyle='#3498db';
    ctx.fillText('● Class -1 (Blue)', 15, 20);
    ctx.fillStyle='#e74c3c';
    ctx.fillText('● Class +1 (Red)', 15, 38);
    ctx.fillStyle='#f39c12';
    ctx.fillText('◎ Support Vectors', 15, 56);
    ctx.fillStyle='#4f46e5';
    ctx.fillText('— Decision Boundary', 15, 74);
  }

  slider.addEventListener('input',()=>{ mVal.textContent=parseFloat(slider.value).toFixed(1); draw(parseFloat(slider.value)); });
  draw(1.5);
})();
</script>
</div>"""


def _html_decision_tree() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Decision Tree — Interactive Traversal</h3>
<p style="text-align:center;color:#555;font-size:13px">Click "Classify Sample" to walk through a decision tree step by step</p>
<svg id="dtSvg" width="520" height="320" style="display:block;margin:auto;overflow:visible"></svg>
<div style="text-align:center;margin-top:8px">
  <button id="dtBtn" style="padding:8px 20px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">Classify Sample</button>
  <button id="dtReset" style="padding:8px 16px;background:#e5e7eb;color:#333;border:none;border-radius:6px;cursor:pointer;font-size:13px;margin-left:8px">Reset</button>
</div>
<div id="dtLog" style="margin:10px auto;max-width:500px;padding:10px;border-radius:8px;background:#f8faff;font-size:13px;min-height:40px;color:#333"></div>
<script>
(function(){
  const svg = document.getElementById('dtSvg');
  const NS = 'http://www.w3.org/2000/svg';

  const nodes = [
    {id:0, x:260, y:40,  label:'Age > 30?',      type:'split'},
    {id:1, x:130, y:130, label:'Income > 50k?',   type:'split'},
    {id:2, x:390, y:130, label:'Credit > 700?',   type:'split'},
    {id:3, x:60,  y:230, label:'REJECT',          type:'leaf', cls:'no'},
    {id:4, x:190, y:230, label:'APPROVE',         type:'leaf', cls:'yes'},
    {id:5, x:320, y:230, label:'APPROVE',         type:'leaf', cls:'yes'},
    {id:6, x:460, y:230, label:'REJECT',          type:'leaf', cls:'no'},
  ];
  const edges = [
    {from:0,to:1,label:'No'},
    {from:0,to:2,label:'Yes'},
    {from:1,to:3,label:'No'},
    {from:1,to:4,label:'Yes'},
    {from:2,to:5,label:'Yes'},
    {from:2,to:6,label:'No'},
  ];

  // Sample paths
  const samples = [
    {vals:{age:25,income:60000,credit:720}, path:[0,1,4], desc:['Age=25 ≤ 30 → No','Income=60k > 50k → Yes','✅ APPROVED']},
    {vals:{age:35,income:45000,credit:680}, path:[0,2,6], desc:['Age=35 > 30 → Yes','Credit=680 ≤ 700 → No','❌ REJECTED']},
    {vals:{age:22,income:30000,credit:800}, path:[0,1,3], desc:['Age=22 ≤ 30 → No','Income=30k ≤ 50k → No','❌ REJECTED']},
    {vals:{age:40,income:80000,credit:750}, path:[0,2,5], desc:['Age=40 > 30 → Yes','Credit=750 > 700 → Yes','✅ APPROVED']},
  ];
  let sIdx = 0, step = 0, active = [];

  function render(){
    svg.innerHTML='';
    // Draw edges
    edges.forEach(e=>{
      const from=nodes[e.from], to=nodes[e.to];
      const line = document.createElementNS(NS,'line');
      line.setAttribute('x1',from.x); line.setAttribute('y1',from.y+22);
      line.setAttribute('x2',to.x);   line.setAttribute('y2',to.y-22);
      line.setAttribute('stroke', active.includes(e.to) && active.includes(e.from) ? '#f59e0b':'#ccc');
      line.setAttribute('stroke-width', active.includes(e.to)&&active.includes(e.from)?2.5:1.5);
      svg.appendChild(line);
      // label
      const tx = document.createElementNS(NS,'text');
      tx.setAttribute('x',(from.x+to.x)/2+6); tx.setAttribute('y',(from.y+to.y)/2);
      tx.setAttribute('font-size','11'); tx.setAttribute('fill','#666'); tx.textContent=e.label;
      svg.appendChild(tx);
    });
    // Draw nodes
    nodes.forEach(n=>{
      const isActive = active.includes(n.id);
      const g = document.createElementNS(NS,'g');
      const rect = document.createElementNS(NS,'rect');
      const w=110, h=40;
      rect.setAttribute('x',n.x-w/2); rect.setAttribute('y',n.y-h/2);
      rect.setAttribute('width',w); rect.setAttribute('height',h);
      rect.setAttribute('rx','8');
      rect.setAttribute('fill', isActive ? (n.cls==='yes'?'#bbf7d0':n.cls==='no'?'#fecaca':'#e0e7ff') : (n.type==='leaf'?(n.cls==='yes'?'#f0fdf4':'#fff5f5'):'#f8faff'));
      rect.setAttribute('stroke', isActive?'#f59e0b':'#c7d2fe');
      rect.setAttribute('stroke-width', isActive?2.5:1.5);
      g.appendChild(rect);
      const txt = document.createElementNS(NS,'text');
      txt.setAttribute('x',n.x); txt.setAttribute('y',n.y+5);
      txt.setAttribute('text-anchor','middle'); txt.setAttribute('font-size','12');
      txt.setAttribute('font-weight', isActive?'bold':'normal');
      txt.setAttribute('fill', isActive?'#92400e':'#1e1b4b');
      txt.textContent = n.label;
      g.appendChild(txt);
      svg.appendChild(g);
    });
  }

  document.getElementById('dtBtn').addEventListener('click',()=>{
    const sample = samples[sIdx % samples.length];
    if(step < sample.path.length){
      active.push(sample.path[step]);
      const log = document.getElementById('dtLog');
      log.innerHTML = '<b>Path so far:</b><br>' + sample.desc.slice(0,step+1).join('<br>');
      step++;
      render();
    }
    if(step >= sample.path.length){ sIdx++; }
  });

  document.getElementById('dtReset').addEventListener('click',()=>{
    active=[]; step=0; render();
    document.getElementById('dtLog').innerHTML='Click "Classify Sample" to start';
  });

  render();
  document.getElementById('dtLog').innerHTML='Click "Classify Sample" to start';
})();
</script>
</div>"""


def _html_neural_network() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Neural Network — Forward Pass Animation</h3>
<p style="text-align:center;color:#555;font-size:13px">Click "Forward Pass" to watch data flow through the network</p>
<canvas id="nnCanvas" width="520" height="300" style="display:block;margin:auto;background:#0f172a;border-radius:10px"></canvas>
<div style="text-align:center;margin-top:10px;display:flex;gap:10px;justify-content:center">
  <button id="nnFwd" style="padding:8px 20px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">▶ Forward Pass</button>
  <button id="nnReset" style="padding:8px 16px;background:#334155;color:#e2e8f0;border:none;border-radius:6px;cursor:pointer;font-size:13px">Reset</button>
</div>
<script>
(function(){
  const canvas = document.getElementById('nnCanvas');
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;

  const layers = [
    {name:'Input',    neurons:3, color:'#818cf8'},
    {name:'Hidden 1', neurons:4, color:'#34d399'},
    {name:'Hidden 2', neurons:4, color:'#f59e0b'},
    {name:'Output',   neurons:2, color:'#f87171'},
  ];
  const xPositions = [70, 200, 340, 470];
  let activationStep = -1;
  let animFrame = null;
  let glowPhase = 0;

  function getNeuronPositions(){
    const positions = [];
    layers.forEach((layer, li)=>{
      const n = layer.neurons;
      const layerPts = [];
      for(let i=0;i<n;i++){
        layerPts.push({x: xPositions[li], y: H/2 + (i-(n-1)/2)*55});
      }
      positions.push(layerPts);
    });
    return positions;
  }

  function draw(step, glow){
    ctx.clearRect(0,0,W,H);
    const positions = getNeuronPositions();

    // Draw connections
    for(let li=0;li<layers.length-1;li++){
      for(let i=0;i<layers[li].neurons;i++){
        for(let j=0;j<layers[li+1].neurons;j++){
          const from = positions[li][i];
          const to   = positions[li+1][j];
          const active = step > li;
          ctx.beginPath();
          ctx.moveTo(from.x,from.y);
          ctx.lineTo(to.x,to.y);
          ctx.strokeStyle = active ? `rgba(129,140,248,${0.3+glow*0.3})` : 'rgba(255,255,255,0.07)';
          ctx.lineWidth = active ? 1.2 : 0.5;
          ctx.stroke();
        }
      }
    }

    // Draw neurons
    positions.forEach((layerPts, li)=>{
      const layer = layers[li];
      const active = step >= li;
      layerPts.forEach(pt=>{
        // Glow
        if(active){
          const grad = ctx.createRadialGradient(pt.x,pt.y,0,pt.x,pt.y,22);
          grad.addColorStop(0, layer.color+'55');
          grad.addColorStop(1,'transparent');
          ctx.beginPath(); ctx.arc(pt.x,pt.y,22,0,Math.PI*2);
          ctx.fillStyle=grad; ctx.fill();
        }
        ctx.beginPath(); ctx.arc(pt.x,pt.y,12,0,Math.PI*2);
        ctx.fillStyle = active ? layer.color : '#1e293b';
        ctx.fill();
        ctx.strokeStyle = active ? '#fff' : '#475569';
        ctx.lineWidth=1.5; ctx.stroke();
        // Activation value
        if(active){
          ctx.fillStyle='#0f172a'; ctx.font='bold 9px sans-serif'; ctx.textAlign='center';
          ctx.fillText((Math.random()*0.8+0.1).toFixed(2), pt.x, pt.y+3);
        }
      });
      // Layer label
      ctx.fillStyle=layer.color; ctx.font='11px sans-serif'; ctx.textAlign='center';
      ctx.fillText(layer.name, xPositions[li], H-8);
    });
  }

  function animate(){
    activationStep++;
    if(activationStep > layers.length) return;
    let phase = 0;
    function frame(){
      phase += 0.08;
      draw(activationStep, Math.sin(phase)*0.5+0.5);
      if(phase < Math.PI*2) animFrame = requestAnimationFrame(frame);
      else if(activationStep < layers.length) animate();
    }
    animFrame = requestAnimationFrame(frame);
  }

  document.getElementById('nnFwd').addEventListener('click',()=>{
    if(animFrame) cancelAnimationFrame(animFrame);
    activationStep = -1;
    animate();
  });
  document.getElementById('nnReset').addEventListener('click',()=>{
    if(animFrame) cancelAnimationFrame(animFrame);
    activationStep = -1;
    draw(-1, 0);
  });

  draw(-1, 0);
})();
</script>
</div>"""


def _html_gradient_descent() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Gradient Descent — Loss Landscape</h3>
<p style="text-align:center;color:#555;font-size:13px">Adjust learning rate and watch the optimizer find the minimum</p>
<canvas id="gdCanvas" width="500" height="300" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="text-align:center;margin-top:10px;display:flex;gap:16px;justify-content:center;flex-wrap:wrap">
  <div>
    <label style="font-size:13px;font-weight:600">Learning Rate: <span id="lrVal">0.1</span></label><br>
    <input type="range" id="lrSlider" min="0.01" max="0.5" step="0.01" value="0.1" style="width:150px">
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <button id="gdStart" style="padding:8px 16px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">▶ Run</button>
    <button id="gdReset" style="padding:8px 14px;background:#e5e7eb;color:#333;border:none;border-radius:6px;cursor:pointer;font-size:13px">Reset</button>
  </div>
</div>
<div id="gdInfo" style="text-align:center;font-size:13px;margin-top:8px;color:#555"></div>
<script>
(function(){
  const canvas = document.getElementById('gdCanvas');
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const pad = {l:50,r:20,t:20,b:40};
  const plotW = W-pad.l-pad.r, plotH = H-pad.t-pad.b;

  // Loss function: 0.5*(x-2)^2 + 0.8*(x-2)^4/20 + small local minima
  function loss(x){ return 0.4*(x-2)**2 + 0.05*(x-6)**2 + Math.sin(x)*0.3; }
  function dloss(x){ return 0.8*(x-2) + 0.1*(x-6) + Math.cos(x)*0.3; }

  const xMin=-2, xMax=10;
  function toCanvas(x,y){
    const yMin=-0.5, yMax=8;
    return {
      cx: pad.l + (x-xMin)/(xMax-xMin)*plotW,
      cy: pad.t + (1-(y-yMin)/(yMax-yMin))*plotH
    };
  }

  let path = [], animId = null, running = false;

  function drawCurve(){
    ctx.clearRect(0,0,W,H);
    // Axes
    ctx.strokeStyle='#ccc'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t+plotH); ctx.lineTo(pad.l+plotW,pad.t+plotH); ctx.stroke();
    // Labels
    ctx.fillStyle='#888'; ctx.font='11px sans-serif'; ctx.textAlign='center';
    ctx.fillText('Parameter θ', pad.l+plotW/2, H-4);
    ctx.save(); ctx.translate(12, pad.t+plotH/2); ctx.rotate(-Math.PI/2);
    ctx.fillText('Loss L(θ)', 0, 0); ctx.restore();

    // Draw loss curve
    ctx.beginPath(); ctx.strokeStyle='#6366f1'; ctx.lineWidth=2.5;
    for(let i=0;i<=200;i++){
      const x = xMin + (xMax-xMin)*i/200;
      const y = loss(x);
      const p = toCanvas(x,y);
      i===0 ? ctx.moveTo(p.cx,p.cy) : ctx.lineTo(p.cx,p.cy);
    }
    ctx.stroke();

    // Global minimum marker
    const minP = toCanvas(2.0, loss(2.0));
    ctx.beginPath(); ctx.arc(minP.cx,minP.cy,5,0,Math.PI*2);
    ctx.fillStyle='#10b981'; ctx.fill();
    ctx.fillStyle='#10b981'; ctx.font='11px sans-serif'; ctx.textAlign='left';
    ctx.fillText('Global Min', minP.cx+7, minP.cy-3);
  }

  function drawPath(){
    if(path.length < 2) return;
    // Trail
    for(let i=1;i<path.length;i++){
      const a=toCanvas(path[i-1][0],path[i-1][1]);
      const b=toCanvas(path[i][0],path[i][1]);
      ctx.beginPath(); ctx.moveTo(a.cx,a.cy); ctx.lineTo(b.cx,b.cy);
      ctx.strokeStyle=`hsla(${40+i*2},90%,55%,0.85)`; ctx.lineWidth=2; ctx.stroke();
      ctx.beginPath(); ctx.arc(a.cx,a.cy,3,0,Math.PI*2);
      ctx.fillStyle='#f59e0b'; ctx.fill();
    }
    // Current position
    const last = path[path.length-1];
    const p = toCanvas(last[0],last[1]);
    ctx.beginPath(); ctx.arc(p.cx,p.cy,7,0,Math.PI*2);
    ctx.fillStyle='#ef4444'; ctx.fill();
    ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();
  }

  function step(){
    if(!path.length) return;
    const lr = parseFloat(document.getElementById('lrSlider').value);
    let [x] = path[path.length-1];
    const grad = dloss(x);
    x = x - lr * grad;
    const y = loss(x);
    path.push([x,y]);
    drawCurve(); drawPath();
    document.getElementById('gdInfo').textContent =
      `Step ${path.length-1} | θ = ${x.toFixed(3)} | Loss = ${y.toFixed(4)} | Gradient = ${grad.toFixed(4)}`;
    if(Math.abs(grad) < 0.01 || path.length > 150){
      cancelAnimationFrame(animId); running=false;
      document.getElementById('gdInfo').textContent += ' ✅ Converged!';
      return;
    }
    animId = requestAnimationFrame(step);
  }

  document.getElementById('lrSlider').addEventListener('input',e=>{
    document.getElementById('lrVal').textContent = parseFloat(e.target.value).toFixed(2);
  });

  document.getElementById('gdStart').addEventListener('click',()=>{
    if(running){ cancelAnimationFrame(animId); running=false; return; }
    const startX = 9.0;
    path = [[startX, loss(startX)]];
    running=true;
    step();
  });
  document.getElementById('gdReset').addEventListener('click',()=>{
    cancelAnimationFrame(animId); running=false; path=[];
    drawCurve();
    document.getElementById('gdInfo').textContent='';
  });

  drawCurve();
})();
</script>
</div>"""


def _html_kmeans() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">K-Means Clustering — Step by Step</h3>
<p style="text-align:center;color:#555;font-size:13px">Watch centroids move to find natural clusters in the data</p>
<canvas id="kmCanvas" width="480" height="340" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa;cursor:pointer"></canvas>
<div style="text-align:center;margin-top:10px;display:flex;gap:8px;justify-content:center">
  <button id="kmStep" style="padding:8px 18px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">Step →</button>
  <button id="kmAuto" style="padding:8px 18px;background:#10b981;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">▶ Auto</button>
  <button id="kmReset" style="padding:8px 14px;background:#e5e7eb;color:#333;border:none;border-radius:6px;cursor:pointer;font-size:13px">Reset</button>
</div>
<div id="kmInfo" style="text-align:center;font-size:13px;margin-top:8px;color:#555;min-height:20px"></div>
<script>
(function(){
  const canvas = document.getElementById('kmCanvas');
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const clrs = ['#e74c3c','#3498db','#2ecc71'];
  let autoId=null;

  // Fixed data
  const rng=(s)=>{ let r=s; return()=>{ r=(r*9301+49297)%233280; return r/233280; }; };
  const rand=rng(99);
  const data=[];
  [[130,240],[340,240],[240,90]].forEach(([cx,cy])=>{
    for(let i=0;i<25;i++) data.push({x:cx+(rand()-0.5)*100, y:cy+(rand()-0.5)*100});
  });

  let centroids = [
    {x:80,y:80},{x:400,y:80},{x:240,y:300}
  ];
  let assignments = data.map(()=>0);
  let iteration = 0;

  function assign(){
    assignments = data.map(p=>{
      let best=0, bestD=Infinity;
      centroids.forEach((c,i)=>{ const d=Math.hypot(p.x-c.x,p.y-c.y); if(d<bestD){bestD=d;best=i;} });
      return best;
    });
  }

  function updateCentroids(){
    centroids = centroids.map((_,ci)=>{
      const pts = data.filter((_,i)=>assignments[i]===ci);
      if(!pts.length) return centroids[ci];
      return {x:pts.reduce((s,p)=>s+p.x,0)/pts.length, y:pts.reduce((s,p)=>s+p.y,0)/pts.length};
    });
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    // Voronoi-like regions (simple coloring)
    for(let px=0;px<W;px+=6){
      for(let py=0;py<H;py+=6){
        let best=0,bestD=Infinity;
        centroids.forEach((c,i)=>{ const d=Math.hypot(px-c.x,py-c.y); if(d<bestD){bestD=d;best=i;} });
        ctx.fillStyle=clrs[best]+'18'; ctx.fillRect(px,py,6,6);
      }
    }
    // Data points
    data.forEach((p,i)=>{
      ctx.beginPath(); ctx.arc(p.x,p.y,5,0,Math.PI*2);
      ctx.fillStyle=clrs[assignments[i]]+'cc'; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.stroke();
    });
    // Centroids
    centroids.forEach((c,i)=>{
      ctx.beginPath(); ctx.arc(c.x,c.y,12,0,Math.PI*2);
      ctx.fillStyle=clrs[i]; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=2.5; ctx.stroke();
      ctx.fillStyle='#fff'; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
      ctx.fillText('C'+(i+1),c.x,c.y+4);
    });
    document.getElementById('kmInfo').textContent = `Iteration ${iteration} — Click "Step" to run next assignment + update`;
  }

  function doStep(){
    assign(); updateCentroids(); iteration++;
    draw();
    document.getElementById('kmInfo').textContent = `Iteration ${iteration} complete`;
  }

  document.getElementById('kmStep').addEventListener('click', doStep);
  document.getElementById('kmAuto').addEventListener('click',()=>{
    if(autoId){ clearInterval(autoId); autoId=null; document.getElementById('kmAuto').textContent='▶ Auto'; return; }
    document.getElementById('kmAuto').textContent='⏸ Pause';
    autoId = setInterval(()=>{ doStep(); if(iteration>=10){ clearInterval(autoId); autoId=null; document.getElementById('kmAuto').textContent='▶ Auto'; }},700);
  });
  document.getElementById('kmReset').addEventListener('click',()=>{
    clearInterval(autoId); autoId=null; iteration=0;
    centroids=[{x:80,y:80},{x:400,y:80},{x:240,y:300}];
    assignments=data.map(()=>0);
    document.getElementById('kmAuto').textContent='▶ Auto';
    draw();
  });

  assign(); draw();
})();
</script>
</div>"""


def _html_overfitting() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Overfitting vs Underfitting</h3>
<p style="text-align:center;color:#555;font-size:13px">Drag the polynomial degree slider to see how model complexity affects fit</p>
<canvas id="ofCanvas" width="500" height="300" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="text-align:center;margin-top:10px">
  <label style="font-size:13px;font-weight:600">Polynomial Degree: <span id="degVal">1</span></label><br>
  <input type="range" id="degSlider" min="1" max="12" step="1" value="1" style="width:220px">
  <div id="ofLabel" style="margin-top:6px;font-size:13px;font-weight:600;color:#ef4444">Underfitting (High Bias)</div>
</div>
<script>
(function(){
  const canvas = document.getElementById('ofCanvas');
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const pad={l:40,r:20,t:20,b:40};
  const plotW=W-pad.l-pad.r, plotH=H-pad.t-pad.b;

  // True function: sin(x)
  const xData=[-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3];
  const yTrue = xData.map(x=>Math.sin(x));
  const noise=[0.15,-0.2,0.1,0.25,-0.15,0.3,0.05,-0.2,0.15,-0.1,0.2,-0.25,0.1];
  const yData = yTrue.map((y,i)=>y+noise[i]);

  function toCanvas(x,y){
    return {
      cx: pad.l+(x-(-3.5))/(7)*plotW,
      cy: pad.t+(1-(y-(-2))/(4))*plotH
    };
  }

  // Simple polynomial fit using least squares (Vandermonde)
  function polyFit(xs, ys, deg){
    const n=xs.length, m=deg+1;
    const A=[], b=ys.slice();
    for(let i=0;i<n;i++){
      const row=[];
      for(let j=0;j<m;j++) row.push(Math.pow(xs[i],j));
      A.push(row);
    }
    // Normal equations: A^T A c = A^T b
    const AT=[], ATA=[], ATb=[];
    for(let j=0;j<m;j++){
      AT.push(A.map(r=>r[j]));
      ATb.push(AT[j].reduce((s,v,i)=>s+v*b[i],0));
    }
    for(let i=0;i<m;i++){
      ATA.push([]);
      for(let j=0;j<m;j++) ATA[i].push(AT[i].reduce((s,v,k)=>s+v*AT[j][k],0));
    }
    // Gaussian elimination
    const mat=ATA.map((r,i)=>[...r,ATb[i]]);
    for(let i=0;i<m;i++){
      let maxR=i;
      for(let r=i+1;r<m;r++) if(Math.abs(mat[r][i])>Math.abs(mat[maxR][i])) maxR=r;
      [mat[i],mat[maxR]]=[mat[maxR],mat[i]];
      for(let r=i+1;r<m;r++){
        const f=mat[r][i]/mat[i][i];
        for(let c=i;c<=m;c++) mat[r][c]-=f*mat[i][c];
      }
    }
    const coeffs=new Array(m).fill(0);
    for(let i=m-1;i>=0;i--){
      coeffs[i]=mat[i][m];
      for(let j=i+1;j<m;j++) coeffs[i]-=mat[i][j]*coeffs[j];
      coeffs[i]/=mat[i][i];
    }
    return coeffs;
  }

  function polyEval(coeffs, x){
    return coeffs.reduce((s,c,i)=>s+c*Math.pow(x,i),0);
  }

  function draw(deg){
    ctx.clearRect(0,0,W,H);
    // Axes
    ctx.strokeStyle='#ddd'; ctx.lineWidth=1;
    const origin=toCanvas(0,-2);
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t+plotH); ctx.lineTo(pad.l+plotW,pad.t+plotH); ctx.stroke();

    // True function (sin)
    ctx.beginPath(); ctx.strokeStyle='#10b981'; ctx.lineWidth=2; ctx.setLineDash([5,4]);
    for(let i=0;i<=100;i++){
      const x=-3.5+7*i/100;
      const p=toCanvas(x,Math.sin(x));
      i===0?ctx.moveTo(p.cx,p.cy):ctx.lineTo(p.cx,p.cy);
    }
    ctx.stroke(); ctx.setLineDash([]);

    // Fitted polynomial
    try{
      const coeffs=polyFit(xData,yData,deg);
      ctx.beginPath(); ctx.strokeStyle='#ef4444'; ctx.lineWidth=2.5;
      let first=true;
      for(let i=0;i<=100;i++){
        const x=-3.5+7*i/100;
        const y=polyEval(coeffs,x);
        if(Math.abs(y)>3) { first=true; continue; }
        const p=toCanvas(x,y);
        first?(ctx.moveTo(p.cx,p.cy),first=false):ctx.lineTo(p.cx,p.cy);
      }
      ctx.stroke();
    }catch(e){}

    // Data points
    xData.forEach((x,i)=>{
      const p=toCanvas(x,yData[i]);
      ctx.beginPath(); ctx.arc(p.cx,p.cy,5,0,Math.PI*2);
      ctx.fillStyle='#6366f1'; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
    });

    // Legend
    ctx.font='11px sans-serif'; ctx.setLineDash([5,4]);
    ctx.strokeStyle='#10b981'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(15,15); ctx.lineTo(35,15); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle='#10b981'; ctx.fillText('True function',38,19);
    ctx.strokeStyle='#ef4444'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(15,33); ctx.lineTo(35,33); ctx.stroke();
    ctx.fillStyle='#ef4444'; ctx.fillText('Fitted model (deg '+deg+')',38,37);
    ctx.beginPath(); ctx.arc(25,51,4,0,Math.PI*2); ctx.fillStyle='#6366f1'; ctx.fill();
    ctx.fillStyle='#6366f1'; ctx.fillText('Training data',38,55);
  }

  const slider=document.getElementById('degSlider');
  const degVal=document.getElementById('degVal');
  const ofLabel=document.getElementById('ofLabel');
  const labels=['','Underfitting (High Bias)','Underfitting (High Bias)','Underfitting','Good Fit','Good Fit','Good Fit','Slightly Overfitting','Overfitting','Overfitting','High Overfitting','High Overfitting','Severe Overfitting'];
  const colors=['','#ef4444','#ef4444','#f59e0b','#10b981','#10b981','#10b981','#f59e0b','#ef4444','#ef4444','#ef4444','#ef4444','#ef4444'];

  slider.addEventListener('input',e=>{
    const d=parseInt(e.target.value);
    degVal.textContent=d;
    ofLabel.textContent=labels[Math.min(d,12)];
    ofLabel.style.color=colors[Math.min(d,12)];
    draw(d);
  });
  draw(1);
})();
</script>
</div>"""


def _html_pca() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Principal Component Analysis (PCA)</h3>
<p style="text-align:center;color:#555;font-size:13px">PCA finds the directions of maximum variance. Click "Project" to reduce to 1D.</p>
<canvas id="pcaCanvas" width="500" height="340" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="text-align:center;margin-top:10px;display:flex;gap:8px;justify-content:center">
  <button id="pcaProject" style="padding:8px 18px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">Project to PC1</button>
  <button id="pcaReset" style="padding:8px 14px;background:#e5e7eb;color:#333;border:none;border-radius:6px;cursor:pointer;font-size:13px">Reset</button>
</div>
<script>
(function(){
  const canvas=document.getElementById('pcaCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;

  // Generate correlated 2D data
  const rng=(s)=>{ let r=s; return()=>{ r=(r*9301+49297)%233280; return r/233280-0.5; }; };
  const rand=rng(7);
  const N=60;
  const raw=[];
  for(let i=0;i<N;i++){
    const t=rand()*6;
    raw.push([t+rand()*1.2, t*0.7+rand()*1.0]);
  }
  // Center
  const mx=raw.reduce((s,p)=>s+p[0],0)/N, my=raw.reduce((s,p)=>s+p[1],0)/N;
  const pts=raw.map(([x,y])=>[x-mx,y-my]);

  // PCA: covariance
  const cxx=pts.reduce((s,p)=>s+p[0]*p[0],0)/N;
  const cyy=pts.reduce((s,p)=>s+p[1]*p[1],0)/N;
  const cxy=pts.reduce((s,p)=>s+p[0]*p[1],0)/N;
  // Eigenvalues
  const tr=cxx+cyy, det=cxx*cyy-cxy*cxy;
  const l1=(tr+Math.sqrt(tr*tr-4*det))/2;
  // PC1 direction
  const pc1 = cxy!==0 ? [l1-cyy, cxy] : [1,0];
  const pc1n = Math.hypot(...pc1);
  const pc1u = pc1.map(v=>v/pc1n);

  const cx=W/2, cy=H/2, scale=35;
  const toC=([x,y])=>({cx:cx+x*scale, cy:cy-y*scale});

  let projected=false, animP=0, animId=null;

  function draw(pFrac=0){
    ctx.clearRect(0,0,W,H);
    // Grid
    ctx.strokeStyle='#eee'; ctx.lineWidth=1;
    for(let x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    // Axes
    ctx.strokeStyle='#ccc'; ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(0,cy);ctx.lineTo(W,cy);ctx.stroke();
    ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,H);ctx.stroke();

    // PC1 axis
    const axLen=200;
    ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2.5; ctx.setLineDash([6,3]);
    ctx.beginPath();
    ctx.moveTo(cx-pc1u[0]*axLen, cy+pc1u[1]*axLen);
    ctx.lineTo(cx+pc1u[0]*axLen, cy-pc1u[1]*axLen);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#f59e0b'; ctx.font='bold 12px sans-serif';
    ctx.fillText('PC1',cx+pc1u[0]*axLen+5, cy-pc1u[1]*axLen-5);

    // Data points (lerp between original and projected)
    pts.forEach(p=>{
      const proj_s = p[0]*pc1u[0]+p[1]*pc1u[1]; // scalar projection
      const ox=p[0], oy=p[1];
      const px2=proj_s*pc1u[0], py2=proj_s*pc1u[1];
      const dx=ox+(px2-ox)*pFrac, dy=oy+(py2-oy)*pFrac;
      const cp=toC([dx,dy]);

      if(pFrac>0.01){
        // projection line
        const orig=toC([ox,oy]);
        ctx.beginPath();ctx.moveTo(orig.cx,orig.cy);ctx.lineTo(cp.cx,cp.cy);
        ctx.strokeStyle='rgba(99,102,241,0.15)'; ctx.lineWidth=1; ctx.stroke();
      }

      ctx.beginPath(); ctx.arc(cp.cx,cp.cy,4.5,0,Math.PI*2);
      ctx.fillStyle=pFrac<0.5?'#6366f1':'#f59e0b';
      ctx.globalAlpha=0.8; ctx.fill(); ctx.globalAlpha=1;
    });

    const varExpl=Math.round(l1/(cxx+cyy)*100);
    ctx.fillStyle='#333'; ctx.font='12px sans-serif'; ctx.textAlign='left';
    ctx.fillText(`PC1 explains ${varExpl}% of variance`, 12, H-12);
  }

  document.getElementById('pcaProject').addEventListener('click',()=>{
    if(animId) return;
    animP=0;
    function frame(){ animP+=0.03; draw(Math.min(animP,1)); if(animP<1) animId=requestAnimationFrame(frame); else animId=null; }
    animId=requestAnimationFrame(frame);
  });
  document.getElementById('pcaReset').addEventListener('click',()=>{
    if(animId){cancelAnimationFrame(animId);animId=null;}
    draw(0);
  });

  draw(0);
})();
</script>
</div>"""


def _html_random_forest() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Random Forest — Ensemble Voting</h3>
<p style="text-align:center;color:#555;font-size:13px">Each tree votes independently; the majority wins. Click "Query Forest"</p>
<svg id="rfSvg" width="520" height="300" style="display:block;margin:auto"></svg>
<div style="text-align:center;margin-top:8px">
  <button id="rfQuery" style="padding:8px 18px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">Query Forest</button>
  <button id="rfReset" style="padding:8px 14px;background:#e5e7eb;color:#333;border:none;border-radius:6px;cursor:pointer;font-size:13px;margin-left:8px">Reset</button>
</div>
<div id="rfResult" style="text-align:center;font-size:13px;margin-top:8px;color:#555;min-height:24px"></div>
<script>
(function(){
  const svg=document.getElementById('rfSvg');
  const NS='http://www.w3.org/2000/svg';
  const W=520, H=300;

  const trees=[
    {x:60,  pred:'🌧 Rain',  conf:0.72, color:'#6366f1'},
    {x:160, pred:'☀️ Sun',   conf:0.61, color:'#f59e0b'},
    {x:260, pred:'🌧 Rain',  conf:0.85, color:'#6366f1'},
    {x:360, pred:'🌧 Rain',  conf:0.68, color:'#6366f1'},
    {x:460, pred:'☀️ Sun',   conf:0.55, color:'#f59e0b'},
  ];

  let step=0;

  function clearSvg(){ svg.innerHTML=''; }

  function drawTree(t, active, voted){
    const g=document.createElementNS(NS,'g');
    // trunk
    const trunk=document.createElementNS(NS,'rect');
    trunk.setAttribute('x',t.x-5); trunk.setAttribute('y',180); trunk.setAttribute('width',10); trunk.setAttribute('height',40);
    trunk.setAttribute('fill','#92400e'); trunk.setAttribute('rx','2');
    g.appendChild(trunk);
    // canopy circles
    [[0,-20,28],[−18,0,22],[18,0,22]].forEach(([dx,dy,r])=>{
      const c=document.createElementNS(NS,'circle');
      c.setAttribute('cx',t.x+dx); c.setAttribute('cy',160+dy); c.setAttribute('r',r);
      c.setAttribute('fill', active?'#16a34a':'#4ade80'); c.setAttribute('opacity', active?'1':'0.4');
      g.appendChild(c);
    });
    // Vote badge
    if(voted){
      const badge=document.createElementNS(NS,'rect');
      badge.setAttribute('x',t.x-32); badge.setAttribute('y',225); badge.setAttribute('width',64); badge.setAttribute('height',22);
      badge.setAttribute('rx','6'); badge.setAttribute('fill',t.color); badge.setAttribute('opacity','0.9');
      g.appendChild(badge);
      const btxt=document.createElementNS(NS,'text');
      btxt.setAttribute('x',t.x); btxt.setAttribute('y',241); btxt.setAttribute('text-anchor','middle');
      btxt.setAttribute('font-size','10'); btxt.setAttribute('fill','#fff'); btxt.setAttribute('font-weight','bold');
      btxt.textContent=t.pred+' '+Math.round(t.conf*100)+'%';
      g.appendChild(btxt);
    }
    // tree number
    const lbl=document.createElementNS(NS,'text');
    lbl.setAttribute('x',t.x); lbl.setAttribute('y',260); lbl.setAttribute('text-anchor','middle');
    lbl.setAttribute('font-size','10'); lbl.setAttribute('fill','#64748b'); lbl.textContent='Tree '+(trees.indexOf(t)+1);
    g.appendChild(lbl);
    svg.appendChild(g);
  }

  function drawAggregator(show){
    if(!show) return;
    const rect=document.createElementNS(NS,'rect');
    rect.setAttribute('x',180); rect.setAttribute('y',8); rect.setAttribute('width',160); rect.setAttribute('height',48);
    rect.setAttribute('rx','10'); rect.setAttribute('fill','#1e1b4b'); rect.setAttribute('stroke','#4f46e5'); rect.setAttribute('stroke-width','2');
    svg.appendChild(rect);
    const txt=document.createElementNS(NS,'text');
    txt.setAttribute('x',260); txt.setAttribute('y',28); txt.setAttribute('text-anchor','middle');
    txt.setAttribute('font-size','12'); txt.setAttribute('fill','#e0e7ff'); txt.setAttribute('font-weight','bold');
    txt.textContent='Majority Vote';
    svg.appendChild(txt);
    const votes={rain:0,sun:0};
    trees.forEach(t=>t.pred.includes('Rain')?votes.rain++:votes.sun++);
    const winner=votes.rain>votes.sun?'🌧 Rain':'☀️ Sun';
    const result=document.createElementNS(NS,'text');
    result.setAttribute('x',260); result.setAttribute('y',48); result.setAttribute('text-anchor','middle');
    result.setAttribute('font-size','13'); result.setAttribute('fill','#a5f3fc'); result.setAttribute('font-weight','bold');
    result.textContent=`${winner} (${Math.max(votes.rain,votes.sun)}/5 votes)`;
    svg.appendChild(result);
    // arrows from trees
    trees.forEach(t=>{
      const line=document.createElementNS(NS,'line');
      line.setAttribute('x1',t.x); line.setAttribute('y1',140);
      line.setAttribute('x2',260); line.setAttribute('y2',56);
      line.setAttribute('stroke','rgba(99,102,241,0.3)'); line.setAttribute('stroke-width','1.5');
      svg.appendChild(line);
    });
  }

  function render(){
    clearSvg();
    drawAggregator(step>=2);
    trees.forEach((t,i)=> drawTree(t, step>=1, step>=1));
  }

  document.getElementById('rfQuery').addEventListener('click',()=>{
    if(step<2){ step++; render(); }
    if(step>=2) document.getElementById('rfResult').textContent='Forest predicts: 🌧 Rain (3 of 5 trees agree)';
  });
  document.getElementById('rfReset').addEventListener('click',()=>{ step=0; render(); document.getElementById('rfResult').textContent=''; });

  render();
})();
</script>
</div>"""


def _html_linear_regression() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Linear Regression — Best Fit Line</h3>
<p style="text-align:center;color:#555;font-size:13px">Drag the slope and intercept to minimize the total squared error</p>
<canvas id="lrCanvas" width="480" height="300" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="display:flex;gap:20px;justify-content:center;margin-top:10px;flex-wrap:wrap">
  <div><label style="font-size:12px;font-weight:600">Slope (m): <span id="mLbl">1.0</span></label><br>
    <input type="range" id="mSlide" min="-2" max="4" step="0.05" value="1.0" style="width:140px"></div>
  <div><label style="font-size:12px;font-weight:600">Intercept (b): <span id="bLbl">0.0</span></label><br>
    <input type="range" id="bSlide" min="-5" max="5" step="0.1" value="0.0" style="width:140px"></div>
  <div id="lrMse" style="display:flex;align-items:center;font-size:13px;color:#555"></div>
</div>
<script>
(function(){
  const canvas=document.getElementById('lrCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;
  const pad={l:45,r:15,t:20,b:35};
  const pW=W-pad.l-pad.r, pH=H-pad.t-pad.b;

  const data=[[1,2.1],[2,3.8],[3,5.2],[4,6.9],[5,8.1],[6,9.4],[7,11.2],[8,12.5],[9,14.1],[10,15.8]].map(([x,y])=>({x,y:y+( Math.sin(x)*0.5)}));
  const xMin=0,xMax=11,yMin=0,yMax=20;

  const toC=(x,y)=>({cx:pad.l+(x-xMin)/(xMax-xMin)*pW, cy:pad.t+(1-(y-yMin)/(yMax-yMin))*pH});

  function mse(m,b){ return data.reduce((s,p)=>{const r=p.y-(m*p.x+b);return s+r*r;},0)/data.length; }

  function draw(m,b){
    ctx.clearRect(0,0,W,H);
    // Axes
    ctx.strokeStyle='#ccc'; ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,pad.t+pH);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pad.l,pad.t+pH);ctx.lineTo(pad.l+pW,pad.t+pH);ctx.stroke();
    // Tick labels
    ctx.fillStyle='#888'; ctx.font='10px sans-serif'; ctx.textAlign='center';
    for(let x=1;x<=10;x++){const p=toC(x,yMin);ctx.fillText(x,p.cx,pad.t+pH+14);}
    ctx.textAlign='right';
    for(let y=0;y<=20;y+=5){const p=toC(xMin,y);ctx.fillText(y,pad.l-5,p.cy+3);}

    // Residual lines
    data.forEach(p=>{
      const pred=m*p.x+b;
      const a=toC(p.x,p.y), b2=toC(p.x,pred);
      ctx.beginPath();ctx.moveTo(a.cx,a.cy);ctx.lineTo(b2.cx,b2.cy);
      ctx.strokeStyle='rgba(239,68,68,0.4)'; ctx.lineWidth=1.5; ctx.stroke();
    });

    // Regression line
    ctx.beginPath(); ctx.strokeStyle='#4f46e5'; ctx.lineWidth=2.5;
    const p0=toC(xMin,m*xMin+b), p1=toC(xMax,m*xMax+b);
    ctx.moveTo(p0.cx,p0.cy); ctx.lineTo(p1.cx,p1.cy); ctx.stroke();

    // Data points
    data.forEach(p=>{
      const cp=toC(p.x,p.y);
      ctx.beginPath(); ctx.arc(cp.cx,cp.cy,5,0,Math.PI*2);
      ctx.fillStyle='#6366f1'; ctx.fill(); ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
    });

    const err=mse(m,b);
    document.getElementById('lrMse').innerHTML=`MSE: <b style="color:${err<2?'#10b981':err<6?'#f59e0b':'#ef4444'}">${err.toFixed(3)}</b>`;
  }

  const mSlide=document.getElementById('mSlide');
  const bSlide=document.getElementById('bSlide');
  function update(){
    const m=parseFloat(mSlide.value), b=parseFloat(bSlide.value);
    document.getElementById('mLbl').textContent=m.toFixed(2);
    document.getElementById('bLbl').textContent=b.toFixed(1);
    draw(m,b);
  }
  mSlide.addEventListener('input',update);
  bSlide.addEventListener('input',update);
  update();
})();
</script>
</div>"""


def _html_logistic_regression() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Logistic Regression — Sigmoid Function</h3>
<p style="text-align:center;color:#555;font-size:13px">The sigmoid squashes any value to a probability between 0 and 1</p>
<canvas id="logCanvas" width="480" height="280" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="text-align:center;margin-top:10px">
  <label style="font-size:12px;font-weight:600">Input value z = <span id="zLbl">0.0</span></label><br>
  <input type="range" id="zSlide" min="-6" max="6" step="0.1" value="0" style="width:200px">
  <div id="sigOut" style="margin-top:8px;font-size:14px;font-weight:600;color:#4f46e5"></div>
</div>
<script>
(function(){
  const canvas=document.getElementById('logCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;
  const pad={l:50,r:20,t:20,b:35};
  const pW=W-pad.l-pad.r, pH=H-pad.t-pad.b;

  const sigmoid=z=>1/(1+Math.exp(-z));
  const zMin=-6,zMax=6,yMin=0,yMax=1;
  const toC=(z,y)=>({cx:pad.l+(z-zMin)/(zMax-zMin)*pW, cy:pad.t+(1-(y-yMin)/(yMax-yMin))*pH});

  function draw(zQuery){
    ctx.clearRect(0,0,W,H);
    // Decision boundary region
    const p05=toC(0,0), p05t=toC(0,1);
    ctx.fillStyle='rgba(239,68,68,0.05)'; ctx.fillRect(pad.l,pad.t,pW/2,pH);
    ctx.fillStyle='rgba(16,185,129,0.05)'; ctx.fillRect(pad.l+pW/2,pad.t,pW/2,pH);
    ctx.strokeStyle='rgba(99,102,241,0.3)'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(toC(0,0).cx,pad.t);ctx.lineTo(toC(0,0).cx,pad.t+pH);ctx.stroke();
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle='#ccc'; ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,pad.t+pH);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pad.l,pad.t+pH);ctx.lineTo(pad.l+pW,pad.t+pH);ctx.stroke();
    // 0.5 line
    const h05=toC(zMin,0.5);
    ctx.strokeStyle='#e5e7eb'; ctx.beginPath();ctx.moveTo(pad.l,h05.cy);ctx.lineTo(pad.l+pW,h05.cy);ctx.stroke();
    ctx.fillStyle='#999'; ctx.font='10px sans-serif'; ctx.textAlign='right'; ctx.fillText('0.5',pad.l-4,h05.cy+3);

    // Labels
    ctx.textAlign='center'; ctx.fillStyle='#888'; ctx.font='10px sans-serif';
    for(let z=-6;z<=6;z+=2){const p=toC(z,0);ctx.fillText(z,p.cx,pad.t+pH+14);}
    ctx.fillText('z (linear score)', pad.l+pW/2, H-2);
    ctx.textAlign='right';
    [0,0.5,1].forEach(y=>{const p=toC(zMin,y);ctx.fillText(y,pad.l-5,p.cy+3);});

    // Sigmoid curve
    ctx.beginPath(); ctx.strokeStyle='#4f46e5'; ctx.lineWidth=2.5;
    for(let i=0;i<=200;i++){
      const z=zMin+(zMax-zMin)*i/200;
      const p=toC(z,sigmoid(z));
      i===0?ctx.moveTo(p.cx,p.cy):ctx.lineTo(p.cx,p.cy);
    }
    ctx.stroke();

    // Query point
    const pQ=toC(zQuery,sigmoid(zQuery));
    // Vertical dashed line from x-axis to point
    ctx.strokeStyle='#f59e0b'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(pQ.cx,pad.t+pH);ctx.lineTo(pQ.cx,pQ.cy);ctx.stroke();
    ctx.beginPath();ctx.moveTo(pad.l,pQ.cy);ctx.lineTo(pQ.cx,pQ.cy);ctx.stroke();
    ctx.setLineDash([]);
    // Point
    ctx.beginPath();ctx.arc(pQ.cx,pQ.cy,7,0,Math.PI*2);
    ctx.fillStyle='#f59e0b'; ctx.fill(); ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();

    // Class label
    const prob=sigmoid(zQuery);
    ctx.fillStyle='#555'; ctx.font='11px sans-serif'; ctx.textAlign='left';
    ctx.fillText(`P(class=1) = σ(${zQuery.toFixed(1)}) = ${prob.toFixed(3)}`,pad.l+4,pad.t+15);
    ctx.fillStyle=zQuery>0?'#10b981':'#ef4444';
    ctx.fillText(zQuery>0?'→ Predict Class 1':'→ Predict Class 0', pad.l+4, pad.t+30);

    ctx.fillStyle='#ef4444'; ctx.font='10px sans-serif'; ctx.textAlign='center';
    ctx.fillText('Class 0 region', pad.l+pW/4, pad.t+pH-8);
    ctx.fillStyle='#10b981';
    ctx.fillText('Class 1 region', pad.l+3*pW/4, pad.t+pH-8);

    document.getElementById('sigOut').textContent=`σ(${zQuery.toFixed(1)}) = ${prob.toFixed(4)} → ${prob>=0.5?'Class 1 ✅':'Class 0 ❌'}`;
  }

  const slider=document.getElementById('zSlide');
  slider.addEventListener('input',()=>{
    const z=parseFloat(slider.value);
    document.getElementById('zLbl').textContent=z.toFixed(1);
    draw(z);
  });
  draw(0);
})();
</script>
</div>"""


def _html_backprop() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Backpropagation — Chain Rule Visualization</h3>
<p style="text-align:center;color:#555;font-size:13px">Watch gradients flow backward through a simple computation graph</p>
<svg id="bpSvg" width="520" height="260" style="display:block;margin:auto"></svg>
<div style="text-align:center;margin-top:8px;display:flex;gap:8px;justify-content:center">
  <button id="bpFwd" style="padding:8px 16px;background:#10b981;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">▶ Forward</button>
  <button id="bpBwd" style="padding:8px 16px;background:#ef4444;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">◀ Backward</button>
  <button id="bpReset" style="padding:8px 14px;background:#e5e7eb;color:#333;border:none;border-radius:6px;cursor:pointer;font-size:13px">Reset</button>
</div>
<div id="bpLog" style="text-align:center;font-size:12px;color:#555;margin-top:8px;padding:6px;background:#f8faff;border-radius:6px;max-width:500px;margin-left:auto;margin-right:auto;min-height:30px"></div>
<script>
(function(){
  const svg=document.getElementById('bpSvg');
  const NS='http://www.w3.org/2000/svg';

  const nodes=[
    {id:'x1',x:40, y:80, label:'x₁=2', val:'2',   grad:'',    color:'#6366f1'},
    {id:'x2',x:40, y:180,label:'x₂=3', val:'3',   grad:'',    color:'#6366f1'},
    {id:'w1',x:40, y:130,label:'w=4',  val:'4',   grad:'',    color:'#f59e0b'},
    {id:'mul',x:160,y:130,label:'×',   val:'',    grad:'',    color:'#8b5cf6'},
    {id:'add',x:280,y:100,label:'+',   val:'',    grad:'',    color:'#8b5cf6'},
    {id:'b',  x:160,y:60, label:'b=1', val:'1',   grad:'',    color:'#f59e0b'},
    {id:'sig',x:400,y:100,label:'σ',   val:'',    grad:'',    color:'#10b981'},
    {id:'L',  x:490,y:100,label:'L',   val:'',    grad:'1',   color:'#ef4444'},
  ];

  const fwdSteps=[
    {node:'mul', val:'8', log:'Forward: × node → w₁·x₁ = 4×2 = 8'},
    {node:'add', val:'9', log:'Forward: + node → mul+b = 8+1 = 9'},
    {node:'sig', val:'0.9999', log:'Forward: σ(9) ≈ 0.9999'},
    {node:'L',   val:'0.9999', log:'Forward: L = σ output = 0.9999'},
  ];
  const bwdSteps=[
    {node:'sig', grad:'0.0001', log:'Backward: ∂L/∂σ = 1; ∂σ/∂z = σ(1-σ) ≈ 0.0001'},
    {node:'add', grad:'0.0001', log:'Backward: gradient passes through + unchanged'},
    {node:'b',   grad:'0.0001', log:'Backward: ∂L/∂b = 0.0001 (bias gradient)'},
    {node:'mul', grad:'0.0001', log:'Backward: gradient reaches × node'},
    {node:'w1',  grad:'0.0002', log:'Backward: ∂L/∂w = grad × x₁ = 0.0001×2 ≈ 0.0002'},
    {node:'x1',  grad:'0.0004', log:'Backward: ∂L/∂x₁ = grad × w = 0.0001×4 = 0.0004'},
  ];

  let phase='idle', fwdIdx=0, bwdIdx=0;
  const nodeState={};
  nodes.forEach(n=>{nodeState[n.id]={val:n.val,grad:n.grad,active:false,bwdActive:false};});

  const edges=[
    ['x1','mul'],['w1','mul'],['mul','add'],['b','add'],['add','sig'],['sig','L']
  ];

  function render(){
    svg.innerHTML='';
    // Edges
    edges.forEach(([a,b])=>{
      const na=nodes.find(n=>n.id===a), nb=nodes.find(n=>n.id===b);
      const fwdActive=nodeState[nb.id]?.active;
      const bwdActive=nodeState[na.id]?.bwdActive;
      const line=document.createElementNS(NS,'line');
      line.setAttribute('x1',na.x+22);line.setAttribute('y1',na.y);
      line.setAttribute('x2',nb.x-22);line.setAttribute('y2',nb.y);
      line.setAttribute('stroke',bwdActive?'#ef4444':fwdActive?'#10b981':'#d1d5db');
      line.setAttribute('stroke-width',bwdActive||fwdActive?2.5:1.5);
      svg.appendChild(line);
      // Arrow
      const arx=nb.x-22, ary=nb.y;
      const arr=document.createElementNS(NS,'polygon');
      arr.setAttribute('points',`${arx},${ary} ${arx-8},${ary-4} ${arx-8},${ary+4}`);
      arr.setAttribute('fill',bwdActive?'#ef4444':fwdActive?'#10b981':'#d1d5db');
      svg.appendChild(arr);
    });

    nodes.forEach(n=>{
      const s=nodeState[n.id];
      const g=document.createElementNS(NS,'g');
      const circ=document.createElementNS(NS,'circle');
      circ.setAttribute('cx',n.x);circ.setAttribute('cy',n.y);circ.setAttribute('r',22);
      circ.setAttribute('fill',s.bwdActive?'#fef2f2':s.active?'#f0fdf4':'#f8faff');
      circ.setAttribute('stroke',s.bwdActive?'#ef4444':s.active?'#10b981':n.color);
      circ.setAttribute('stroke-width',s.active||s.bwdActive?2.5:1.5);
      g.appendChild(circ);
      // Node label
      const lbl=document.createElementNS(NS,'text');
      lbl.setAttribute('x',n.x);lbl.setAttribute('y',n.y-2);lbl.setAttribute('text-anchor','middle');
      lbl.setAttribute('font-size','11');lbl.setAttribute('font-weight','bold');lbl.setAttribute('fill',n.color);
      lbl.textContent=n.label;
      g.appendChild(lbl);
      // Value
      if(s.val){
        const vt=document.createElementNS(NS,'text');
        vt.setAttribute('x',n.x);vt.setAttribute('y',n.y+10);vt.setAttribute('text-anchor','middle');
        vt.setAttribute('font-size','9');vt.setAttribute('fill','#10b981');
        vt.textContent='='+s.val;
        g.appendChild(vt);
      }
      // Grad
      if(s.grad){
        const gt=document.createElementNS(NS,'text');
        gt.setAttribute('x',n.x);gt.setAttribute('y',n.y+24);gt.setAttribute('text-anchor','middle');
        gt.setAttribute('font-size','8');gt.setAttribute('fill','#ef4444');
        gt.textContent='∂='+s.grad;
        g.appendChild(gt);
      }
      svg.appendChild(g);
    });
  }

  document.getElementById('bpFwd').addEventListener('click',()=>{
    if(fwdIdx>=fwdSteps.length) return;
    const s=fwdSteps[fwdIdx++];
    nodeState[s.node].val=s.val; nodeState[s.node].active=true;
    document.getElementById('bpLog').textContent=s.log;
    render();
  });
  document.getElementById('bpBwd').addEventListener('click',()=>{
    if(bwdIdx>=bwdSteps.length) return;
    const s=bwdSteps[bwdIdx++];
    nodeState[s.node].grad=s.grad; nodeState[s.node].bwdActive=true;
    document.getElementById('bpLog').textContent=s.log;
    render();
  });
  document.getElementById('bpReset').addEventListener('click',()=>{
    fwdIdx=0; bwdIdx=0;
    nodes.forEach(n=>{nodeState[n.id]={val:n.val,grad:n.grad,active:false,bwdActive:false};});
    document.getElementById('bpLog').textContent='';
    render();
  });

  render();
})();
</script>
</div>"""


def _html_attention() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Self-Attention Mechanism</h3>
<p style="text-align:center;color:#555;font-size:13px">Click a word to see which tokens it attends to most</p>
<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
  <div id="attnViz" style="position:relative;width:380px;height:300px;border:1px solid #e0e7ff;border-radius:10px;background:#fafbff;overflow:hidden">
    <canvas id="attnCanvas" width="380" height="300" style="position:absolute;top:0;left:0"></canvas>
    <div id="attnWords" style="position:absolute;top:0;left:0;width:100%;height:100%"></div>
  </div>
  <div style="min-width:160px;max-width:200px">
    <div style="font-size:12px;font-weight:600;margin-bottom:8px;color:#4f46e5">Attention weights:</div>
    <div id="attnBars"></div>
    <div style="margin-top:12px;font-size:11px;color:#888">In transformers, each word creates Q, K, V vectors. Attention = softmax(QKᵀ/√d)·V</div>
  </div>
</div>
<script>
(function(){
  const canvas=document.getElementById('attnCanvas');
  const ctx=canvas.getContext('2d');
  const W=380,H=300;

  const sentence=['The','cat','sat','on','the','mat'];
  // Fake attention matrix (rows=query, cols=key)
  const attnMatrix=[
    [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.05,0.6, 0.2, 0.05,0.05,0.05],
    [0.05,0.3, 0.45,0.1, 0.05,0.05],
    [0.1, 0.05,0.1, 0.4, 0.2, 0.15],
    [0.1, 0.1, 0.05,0.1, 0.5, 0.15],
    [0.05,0.1, 0.2, 0.1, 0.1, 0.45],
  ];

  const N=sentence.length;
  const xPositions=sentence.map((_,i)=>30+i*(320/N)+(320/N)/2);
  const yTop=60, yBot=240;

  let selected=null;

  function drawLines(qi){
    ctx.clearRect(0,0,W,H);
    if(qi===null) return;
    attnMatrix[qi].forEach((w,ki)=>{
      if(w<0.05) return;
      ctx.beginPath();
      ctx.moveTo(xPositions[qi],yTop+20);
      ctx.lineTo(xPositions[ki],yBot-20);
      ctx.strokeStyle=`rgba(99,102,241,${w*1.5})`;
      ctx.lineWidth=w*14;
      ctx.stroke();
    });
  }

  function renderWords(){
    const div=document.getElementById('attnWords');
    div.innerHTML='';
    sentence.forEach((w,i)=>{
      const makeWord=(y,isQuery)=>{
        const span=document.createElement('div');
        span.textContent=w;
        span.style.cssText=`position:absolute;left:${xPositions[i]-25}px;top:${y}px;width:50px;text-align:center;
          padding:5px 4px;border-radius:6px;font-size:13px;cursor:pointer;font-weight:600;
          background:${selected===i&&isQuery?'#4f46e5':'#e0e7ff'};
          color:${selected===i&&isQuery?'#fff':'#1e1b4b'};
          border:2px solid ${selected===i&&isQuery?'#4f46e5':'transparent'};
          transition:all 0.2s;`;
        if(isQuery) span.addEventListener('click',()=>{
          selected=(selected===i?null:i);
          drawLines(selected);
          renderWords();
          renderBars(selected);
        });
        return span;
      };
      div.appendChild(makeWord(yTop-10, true));
      div.appendChild(makeWord(yBot-10, false));
    });
    // Labels
    const ql=document.createElement('div');
    ql.style.cssText='position:absolute;left:4px;top:48px;font-size:10px;color:#6366f1;font-weight:600;writing-mode:initial';
    ql.textContent='Query →'; div.appendChild(ql);
    const kl=document.createElement('div');
    kl.style.cssText='position:absolute;left:4px;top:228px;font-size:10px;color:#10b981;font-weight:600';
    kl.textContent='Keys →'; div.appendChild(kl);
  }

  function renderBars(qi){
    const div=document.getElementById('attnBars');
    if(qi===null){div.innerHTML='<p style="color:#aaa;font-size:12px">Select a word above</p>';return;}
    div.innerHTML=sentence.map((w,i)=>`
      <div style="margin-bottom:6px">
        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">
          <span>${w}</span><span>${(attnMatrix[qi][i]*100).toFixed(0)}%</span>
        </div>
        <div style="background:#e0e7ff;border-radius:4px;height:8px">
          <div style="background:#4f46e5;border-radius:4px;height:8px;width:${attnMatrix[qi][i]*100}%;transition:width 0.3s"></div>
        </div>
      </div>
    `).join('');
  }

  drawLines(null); renderWords(); renderBars(null);
})();
</script>
</div>"""


def _html_regularization() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">L1 vs L2 Regularization</h3>
<p style="text-align:center;color:#555;font-size:13px">Adjust λ to see how regularization shrinks model weights</p>
<canvas id="regCanvas" width="500" height="280" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="text-align:center;margin-top:10px;display:flex;gap:20px;justify-content:center;flex-wrap:wrap">
  <div>
    <label style="font-size:12px;font-weight:600">λ (strength): <span id="lambdaLbl">0.1</span></label><br>
    <input type="range" id="lambdaSlide" min="0" max="1" step="0.01" value="0.1" style="width:160px">
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <label style="font-size:12px"><input type="radio" name="regType" value="none" style="margin-right:4px">None</label>
    <label style="font-size:12px"><input type="radio" name="regType" value="l1" style="margin-right:4px">L1 (Lasso)</label>
    <label style="font-size:12px"><input type="radio" name="regType" value="l2" checked style="margin-right:4px">L2 (Ridge)</label>
  </div>
</div>
<div id="regInfo" style="text-align:center;font-size:12px;color:#555;margin-top:8px"></div>
<script>
(function(){
  const canvas=document.getElementById('regCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;
  const pad={l:50,r:20,t:20,b:35};
  const pW=W-pad.l-pad.r, pH=H-pad.t-pad.b;

  // 10 weights before regularization
  const rawWeights=[2.1,-1.8,3.2,-0.5,1.4,-2.9,0.3,1.7,-1.2,2.5];
  const wMax=4;

  function applyReg(w, lambda, type){
    if(type==='none') return w;
    if(type==='l1'){
      // Soft thresholding
      const sign = w>0?1:-1;
      return Math.max(0,Math.abs(w)-lambda)*sign;
    }
    // L2: scale down
    return w/(1+lambda*2);
  }

  function toC(x,y){
    return {cx:pad.l+x/rawWeights.length*pW, cy:pad.t+(1-(y+wMax)/(2*wMax))*pH};
  }

  function draw(lambda, type){
    ctx.clearRect(0,0,W,H);
    // Zero line
    const zeroY=pad.t+pH/2;
    ctx.strokeStyle='#ccc'; ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.l,zeroY);ctx.lineTo(pad.l+pW,zeroY);ctx.stroke();
    ctx.fillStyle='#aaa';ctx.font='10px sans-serif';ctx.textAlign='right';ctx.fillText('0',pad.l-4,zeroY+3);

    const barW = pW/rawWeights.length*0.35;

    rawWeights.forEach((w,i)=>{
      const rw=applyReg(w,lambda,type);
      const x=(i+0.5)/rawWeights.length*pW;

      // Original bar (faded)
      const oTop=toC(0,w), oBot=toC(0,0);
      ctx.fillStyle=w>0?'rgba(99,102,241,0.2)':'rgba(239,68,68,0.2)';
      const oh=Math.abs(oTop.cy-zeroY);
      ctx.fillRect(pad.l+x-barW-2, Math.min(oTop.cy,zeroY), barW*1.5, oh+0.5);

      // Regularized bar
      const rTop=pad.t+(1-(rw+wMax)/(2*wMax))*pH;
      ctx.fillStyle=rw>0?'rgba(99,102,241,0.85)':'rgba(239,68,68,0.85)';
      const rh=Math.abs(rTop-zeroY);
      ctx.fillRect(pad.l+x-barW/2, Math.min(rTop,zeroY), barW, Math.max(rh,1));

      // Label
      ctx.fillStyle='#555'; ctx.font='9px sans-serif'; ctx.textAlign='center';
      ctx.fillText('w'+(i+1), pad.l+x, pad.t+pH+14);

      // Sparse indicator for L1
      if(type==='l1' && Math.abs(rw)<0.01){
        ctx.fillStyle='#10b981'; ctx.font='bold 10px sans-serif';
        ctx.fillText('0', pad.l+x, zeroY-8);
      }
    });

    // Legend
    ctx.font='11px sans-serif'; ctx.textAlign='left';
    ctx.fillStyle='rgba(99,102,241,0.4)'; ctx.fillRect(15,12,14,12);
    ctx.fillStyle='#555'; ctx.fillText('Original', 32, 22);
    ctx.fillStyle='rgba(99,102,241,0.85)'; ctx.fillRect(15,28,14,12);
    ctx.fillStyle='#555'; ctx.fillText('After regularization', 32, 38);

    // Stats
    const nZero=rawWeights.filter(w=>Math.abs(applyReg(w,lambda,type))<0.01).length;
    const l1norm=rawWeights.reduce((s,w)=>s+Math.abs(applyReg(w,lambda,type)),0);
    document.getElementById('regInfo').innerHTML =
      type==='l1'
        ? `L1: <b>${nZero}/10 weights zeroed out</b> (sparse solution) | ‖w‖₁ = ${l1norm.toFixed(2)}`
        : type==='l2'
        ? `L2: All weights shrunk proportionally | ‖w‖₂ = ${Math.sqrt(rawWeights.reduce((s,w)=>s+applyReg(w,lambda,type)**2,0)).toFixed(2)}`
        : 'No regularization applied';
  }

  function update(){
    const lambda=parseFloat(document.getElementById('lambdaSlide').value);
    const type=document.querySelector('input[name="regType"]:checked').value;
    document.getElementById('lambdaLbl').textContent=lambda.toFixed(2);
    draw(lambda,type);
  }
  document.getElementById('lambdaSlide').addEventListener('input',update);
  document.querySelectorAll('input[name="regType"]').forEach(r=>r.addEventListener('change',update));
  update();
})();
</script>
</div>"""


def _html_dbscan() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">DBSCAN — Density-Based Clustering</h3>
<p style="text-align:center;color:#555;font-size:13px">Adjust ε (epsilon) to see how neighborhoods form clusters and detect noise</p>
<canvas id="dbCanvas" width="480" height="320" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="text-align:center;margin-top:10px">
  <label style="font-size:12px;font-weight:600">ε (radius): <span id="epsLbl">30</span>px  |  minPts: 3</label><br>
  <input type="range" id="epsSlide" min="10" max="80" step="2" value="30" style="width:200px">
</div>
<div id="dbInfo" style="text-align:center;font-size:13px;margin-top:8px;color:#555"></div>
<script>
(function(){
  const canvas=document.getElementById('dbCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;

  const rng=(s)=>{let r=s;return()=>{r=(r*9301+49297)%233280;return r/233280;};};
  const rand=rng(13);
  const pts=[];
  // Cluster A
  [[120,150],[280,100],[380,220],[150,260]].forEach(([cx,cy])=>{
    for(let i=0;i<12;i++) pts.push({x:cx+(rand()-0.5)*55, y:cy+(rand()-0.5)*55});
  });
  // Noise
  for(let i=0;i<8;i++) pts.push({x:rand()*W*0.9+20, y:rand()*H*0.9+20});

  const minPts=3;
  const clusterColors=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6'];

  function dbscan(eps){
    const labels=new Array(pts.length).fill(-2); // -2=unvisited, -1=noise
    let clusterId=0;
    function regionQuery(i){
      return pts.reduce((nb,p,j)=>{ if(Math.hypot(pts[i].x-p.x,pts[i].y-p.y)<=eps) nb.push(j); return nb; },[]);
    }
    function expand(i,neighbors,cid){
      labels[i]=cid;
      let qi=[...neighbors];
      while(qi.length){
        const j=qi.shift();
        if(labels[j]===-2){
          labels[j]=cid;
          const nb2=regionQuery(j);
          if(nb2.length>=minPts) qi=qi.concat(nb2.filter(k=>!qi.includes(k)));
        }
        if(labels[j]===-2||labels[j]===-1) labels[j]=cid;
      }
    }
    pts.forEach((_,i)=>{
      if(labels[i]!==-2) return;
      const nb=regionQuery(i);
      if(nb.length<minPts){ labels[i]=-1; return; }
      expand(i,nb,clusterId++);
    });
    return labels;
  }

  function draw(eps){
    ctx.clearRect(0,0,W,H);
    const labels=dbscan(eps);
    const nClusters=Math.max(...labels)+1;

    pts.forEach((p,i)=>{
      // Epsilon circle (faint)
      ctx.beginPath(); ctx.arc(p.x,p.y,eps,0,Math.PI*2);
      ctx.strokeStyle='rgba(200,200,200,0.2)'; ctx.lineWidth=0.5; ctx.stroke();

      const isNoise=labels[i]===-1;
      ctx.beginPath(); ctx.arc(p.x,p.y, isNoise?4:6, 0,Math.PI*2);
      ctx.fillStyle=isNoise?'#94a3b8':clusterColors[labels[i]%clusterColors.length];
      ctx.fill();
      if(!isNoise){ ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke(); }
    });

    const nNoise=labels.filter(l=>l===-1).length;
    document.getElementById('dbInfo').innerHTML=
      `Found <b>${nClusters} cluster${nClusters!==1?'s':''}</b> | <b style="color:#94a3b8">${nNoise} noise points</b>`;
  }

  const slider=document.getElementById('epsSlide');
  slider.addEventListener('input',()=>{
    const eps=parseInt(slider.value);
    document.getElementById('epsLbl').textContent=eps;
    draw(eps);
  });
  draw(30);
})();
</script>
</div>"""


def _html_cnn() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">Convolutional Neural Network (CNN)</h3>
<p style="text-align:center;color:#555;font-size:13px">Click "Convolve" to see how a filter slides across the input to create a feature map</p>
<canvas id="cnnCanvas" width="500" height="280" style="display:block;margin:auto;border:1px solid #e0e7ff;border-radius:10px;background:#0f172a"></canvas>
<div style="text-align:center;margin-top:10px;display:flex;gap:8px;justify-content:center">
  <button id="cnnStep" style="padding:8px 18px;background:#6366f1;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">Convolve →</button>
  <button id="cnnReset" style="padding:8px 14px;background:#334155;color:#e2e8f0;border:none;border-radius:6px;cursor:pointer;font-size:13px">Reset</button>
</div>
<div id="cnnLog" style="text-align:center;font-size:12px;color:#94a3b8;margin-top:8px;min-height:20px"></div>
<script>
(function(){
  const canvas=document.getElementById('cnnCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;

  // 6x6 input
  const input=[
    [0,0,1,1,0,0],
    [0,1,1,1,1,0],
    [1,1,0,0,1,1],
    [1,1,0,0,1,1],
    [0,1,1,1,1,0],
    [0,0,1,1,0,0],
  ];
  // Edge-detection filter (3x3)
  const filter=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]];

  const cellSz=28, inputX=30, inputY=40;
  const filterX=230, filterY=40;
  const outX=370, outY=40;

  let positions=[]; // [row,col] positions for conv steps
  for(let r=0;r<=3;r++) for(let c=0;c<=3;c++) positions.push([r,c]);
  let step=0;
  const featureMap=Array.from({length:4},()=>Array(4).fill(null));

  function conv(r,c){
    let sum=0;
    for(let fr=0;fr<3;fr++) for(let fc=0;fc<3;fc++) sum+=input[r+fr][c+fc]*filter[fr][fc];
    return sum;
  }

  function draw(){
    ctx.clearRect(0,0,W,H);

    // Input grid
    ctx.fillStyle='#1e293b'; ctx.font='10px sans-serif'; ctx.textAlign='center';
    ctx.fillStyle='#818cf8'; ctx.fillText('Input (6×6)',inputX+3*cellSz,inputY-10);
    input.forEach((row,r)=>row.forEach((v,c)=>{
      const x=inputX+c*cellSz, y=inputY+r*cellSz;
      ctx.fillStyle=v?`rgba(129,140,248,${0.3+v*0.6})`:'#1e293b';
      ctx.fillRect(x,y,cellSz-2,cellSz-2);
      ctx.strokeStyle='#334155'; ctx.lineWidth=1; ctx.strokeRect(x,y,cellSz-2,cellSz-2);
      ctx.fillStyle=v?'#fff':'#475569'; ctx.font='10px sans-serif'; ctx.textAlign='center';
      ctx.fillText(v,x+cellSz/2-1,y+cellSz/2+4);
    }));

    // Filter
    ctx.fillStyle='#f59e0b'; ctx.textAlign='center'; ctx.font='10px sans-serif';
    ctx.fillText('Filter (3×3)',filterX+1.5*cellSz,inputY-10);
    filter.forEach((row,r)=>row.forEach((v,c)=>{
      const x=filterX+c*cellSz, y=filterY+r*cellSz;
      ctx.fillStyle=v>0?'rgba(245,158,11,0.5)':'rgba(239,68,68,0.4)';
      ctx.fillRect(x,y,cellSz-2,cellSz-2);
      ctx.strokeStyle='#475569'; ctx.lineWidth=1; ctx.strokeRect(x,y,cellSz-2,cellSz-2);
      ctx.fillStyle='#fff'; ctx.font='9px sans-serif';
      ctx.fillText(v,x+cellSz/2-1,y+cellSz/2+3);
    }));

    // Feature map
    ctx.fillStyle='#34d399'; ctx.textAlign='center'; ctx.font='10px sans-serif';
    ctx.fillText('Feature Map (4×4)',outX+2*cellSz,inputY-10);
    for(let r=0;r<4;r++) for(let c=0;c<4;c++){
      const x=outX+c*cellSz, y=outY+r*cellSz;
      const v=featureMap[r][c];
      ctx.fillStyle=v!==null?`rgba(52,211,153,${Math.min(Math.abs(v)/8,1)*0.8+0.1})`:'#1e293b';
      ctx.fillRect(x,y,cellSz-2,cellSz-2);
      ctx.strokeStyle='#334155'; ctx.lineWidth=1; ctx.strokeRect(x,y,cellSz-2,cellSz-2);
      if(v!==null){ ctx.fillStyle='#fff'; ctx.font='9px sans-serif'; ctx.fillText(v,x+cellSz/2-1,y+cellSz/2+3); }
    }

    // Highlight current position on input
    if(step>0 && step<=positions.length){
      const [r,c]=positions[step-1];
      ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2.5;
      ctx.strokeRect(inputX+c*cellSz-1,inputY+r*cellSz-1,3*cellSz,3*cellSz);
    }

    // Labels
    ctx.fillStyle='#94a3b8'; ctx.font='11px sans-serif'; ctx.textAlign='center';
    ctx.fillText('×', filterX-16, inputY+50);
    ctx.fillText('=', outX-16, inputY+50);
  }

  document.getElementById('cnnStep').addEventListener('click',()=>{
    if(step>=positions.length) return;
    const [r,c]=positions[step];
    featureMap[r][c]=conv(r,c);
    step++;
    draw();
    document.getElementById('cnnLog').textContent=
      `Step ${step}/${positions.length}: Filter at (${positions[step-1][0]},${positions[step-1][1]}) → output = ${featureMap[positions[step-1][0]][positions[step-1][1]]}`;
  });
  document.getElementById('cnnReset').addEventListener('click',()=>{
    step=0;
    for(let r=0;r<4;r++) for(let c=0;c<4;c++) featureMap[r][c]=null;
    document.getElementById('cnnLog').textContent='';
    draw();
  });
  draw();
})();
</script>
</div>"""


def _html_rnn() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">RNN / LSTM — Sequential Memory</h3>
<p style="text-align:center;color:#555;font-size:13px">Watch how the hidden state carries memory across time steps</p>
<canvas id="rnnCanvas" width="520" height="260" style="display:block;margin:auto;border:1px solid #e0e7ff;border-radius:10px;background:#0f172a"></canvas>
<div style="text-align:center;margin-top:10px;display:flex;gap:8px;justify-content:center">
  <button id="rnnStep" style="padding:8px 18px;background:#6366f1;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">Next Token →</button>
  <button id="rnnReset" style="padding:8px 14px;background:#334155;color:#e2e8f0;border:none;border-radius:6px;cursor:pointer;font-size:13px">Reset</button>
</div>
<div id="rnnLog" style="text-align:center;font-size:12px;color:#94a3b8;margin-top:8px;padding:6px;min-height:20px"></div>
<script>
(function(){
  const canvas=document.getElementById('rnnCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;

  const tokens=['The','cat','sat','on','mat'];
  const hiddenStates=[
    [0.1,0.0,0.0,0.2],
    [0.5,0.3,0.1,0.4],
    [0.6,0.7,0.2,0.5],
    [0.4,0.6,0.8,0.3],
    [0.3,0.5,0.9,0.7],
  ];
  const outputs=['h₁','h₂','h₃','h₄','h₅'];

  let step=0;
  const cellXs=tokens.map((_,i)=>70+i*90);
  const cellY=120;

  function drawCell(x,y,label,active,hs){
    // Main cell
    const w=64,h=50;
    ctx.fillStyle=active?'#312e81':'#1e293b';
    ctx.strokeStyle=active?'#818cf8':'#334155';
    ctx.lineWidth=active?2:1;
    ctx.beginPath(); ctx.roundRect(x-w/2,y-h/2,w,h,8); ctx.fill(); ctx.stroke();
    // Label
    ctx.fillStyle=active?'#e0e7ff':'#64748b'; ctx.font=`bold 12px sans-serif`; ctx.textAlign='center';
    ctx.fillText(label,x,y+4);
    // Hidden state bars
    if(hs && active){
      hs.forEach((v,i)=>{
        ctx.fillStyle=`rgba(129,140,248,${0.3+v*0.7})`;
        ctx.fillRect(x-w/2+2+i*(w/hs.length-1), y+h/2+4, w/hs.length-3, 8);
      });
    }
  }

  function draw(){
    ctx.clearRect(0,0,W,H);

    // Input tokens
    tokens.forEach((t,i)=>{
      const active=i<step;
      const x=cellXs[i];
      ctx.fillStyle=active?'#1d4ed8':'#1e293b';
      ctx.strokeStyle=active?'#60a5fa':'#334155';
      ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.roundRect(x-28,200,56,30,6); ctx.fill(); ctx.stroke();
      ctx.fillStyle=active?'#bfdbfe':'#475569'; ctx.font='11px sans-serif'; ctx.textAlign='center';
      ctx.fillText(t,x,220);

      // Arrow up
      if(active){
        ctx.strokeStyle='#60a5fa'; ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.moveTo(x,200); ctx.lineTo(x,cellY+25); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x-4,cellY+28); ctx.lineTo(x,cellY+22); ctx.lineTo(x+4,cellY+28); ctx.stroke();
      }
    });

    // RNN cells
    tokens.forEach((t,i)=>{
      drawCell(cellXs[i],cellY,`t=${i+1}`,i<step,i<step?hiddenStates[i]:null);
    });

    // Hidden state arrows between cells
    for(let i=0;i<step-1;i++){
      ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2;
      ctx.beginPath(); ctx.moveTo(cellXs[i]+32,cellY); ctx.lineTo(cellXs[i+1]-32,cellY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cellXs[i+1]-35,cellY-4); ctx.lineTo(cellXs[i+1]-29,cellY); ctx.lineTo(cellXs[i+1]-35,cellY+4); ctx.fill();
      ctx.fillStyle='#f59e0b'; ctx.font='9px sans-serif'; ctx.textAlign='center';
      ctx.fillText('h',cellXs[i]+50,cellY-8);
    }

    // Output arrows
    for(let i=0;i<step;i++){
      ctx.strokeStyle='#34d399'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(cellXs[i],cellY-25); ctx.lineTo(cellXs[i],50); ctx.stroke();
      ctx.fillStyle='#064e3b'; ctx.strokeStyle='#34d399'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.roundRect(cellXs[i]-18,28,36,22,5); ctx.fill(); ctx.stroke();
      ctx.fillStyle='#6ee7b7'; ctx.font='10px sans-serif'; ctx.textAlign='center';
      ctx.fillText(outputs[i],cellXs[i],44);
    }

    // Labels
    ctx.fillStyle='#60a5fa'; ctx.font='10px sans-serif'; ctx.textAlign='left';
    ctx.fillText('xₜ (inputs)', 8, 222);
    ctx.fillStyle='#f59e0b'; ctx.fillText('hₜ (hidden state)', 8, cellY+2);
    ctx.fillStyle='#34d399'; ctx.fillText('yₜ (outputs)', 8, 42);
  }

  document.getElementById('rnnStep').addEventListener('click',()=>{
    if(step>=tokens.length) return;
    step++;
    draw();
    document.getElementById('rnnLog').textContent=
      `Processing "${tokens[step-1]}" → hidden state updated with memory from previous ${step-1} token${step-1!==1?'s':''}`;
  });
  document.getElementById('rnnReset').addEventListener('click',()=>{
    step=0; draw(); document.getElementById('rnnLog').textContent='';
  });
  draw();
})();
</script>
</div>"""


def _html_tsne() -> str:
    return """
<div style="font-family:sans-serif;padding:12px">
<h3 style="text-align:center;color:#4f46e5">t-SNE — Dimensionality Reduction</h3>
<p style="text-align:center;color:#555;font-size:13px">t-SNE iteratively arranges high-dimensional points to preserve local structure</p>
<canvas id="tsneCanvas" width="480" height="320" style="display:block;margin:auto;border:1px solid #ddd;border-radius:8px;background:#fafafa"></canvas>
<div style="text-align:center;margin-top:10px;display:flex;gap:8px;justify-content:center">
  <button id="tsneRun" style="padding:8px 18px;background:#4f46e5;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px">▶ Animate t-SNE</button>
  <button id="tsneReset" style="padding:8px 14px;background:#e5e7eb;color:#333;border:none;border-radius:6px;cursor:pointer;font-size:13px">Reset</button>
</div>
<div id="tsneInfo" style="text-align:center;font-size:12px;color:#555;margin-top:8px">Click animate to watch high-dimensional clusters emerge in 2D</div>
<script>
(function(){
  const canvas=document.getElementById('tsneCanvas');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;
  const clrs=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6'];

  const rng=(s)=>{let r=s;return()=>{r=(r*9301+49297)%233280;return r/233280;};};
  const rand=rng(7);
  const N=60, K=4;

  // True cluster centers (in 2D for simplicity)
  const centers=[[0.2,0.2],[0.8,0.2],[0.2,0.8],[0.8,0.8]];

  // Initial random positions
  let pts=Array.from({length:N},(_,i)=>{
    const cls=i%K;
    return {cls, x:rand(), y:rand(), // random start
            tx:centers[cls][0]+(rand()-0.5)*0.15, // true 2D
            ty:centers[cls][1]+(rand()-0.5)*0.15};
  });

  let animId=null, t=0;

  function lerp(a,b,f){return a+(b-a)*f;}

  function draw(frac){
    ctx.clearRect(0,0,W,H);
    const pad=40;
    pts.forEach(p=>{
      const x=pad+(lerp(p.x,p.tx,frac))*(W-2*pad);
      const y=pad+(lerp(p.y,p.ty,frac))*(H-2*pad);
      ctx.beginPath(); ctx.arc(x,y,6,0,Math.PI*2);
      ctx.fillStyle=clrs[p.cls]; ctx.globalAlpha=0.8; ctx.fill();
      ctx.globalAlpha=1; ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
    });
    // Legend
    clrs.slice(0,K).forEach((c,i)=>{
      ctx.beginPath(); ctx.arc(16,16+i*18,5,0,Math.PI*2);
      ctx.fillStyle=c; ctx.fill();
      ctx.fillStyle='#555'; ctx.font='11px sans-serif'; ctx.textAlign='left';
      ctx.fillText('Class '+(i+1),25,20+i*18);
    });
    document.getElementById('tsneInfo').textContent=
      frac<0.05?'Random initialization — no structure yet':
      frac<0.5?`Iterating... clusters emerging (${Math.round(frac*100)}%)`:
      frac<0.99?`Fine-tuning cluster separation (${Math.round(frac*100)}%)`:
      '✅ Converged! Similar points cluster together in 2D';
  }

  document.getElementById('tsneRun').addEventListener('click',()=>{
    if(animId){cancelAnimationFrame(animId);animId=null;return;}
    t=0;
    function frame(){
      t+=0.008;
      // Ease in-out
      const f=t<1?t*t*(3-2*t):1;
      draw(f);
      if(t<1.05) animId=requestAnimationFrame(frame); else animId=null;
    }
    animId=requestAnimationFrame(frame);
  });
  document.getElementById('tsneReset').addEventListener('click',()=>{
    if(animId){cancelAnimationFrame(animId);animId=null;}
    t=0; draw(0);
    document.getElementById('tsneInfo').textContent='Click animate to watch high-dimensional clusters emerge in 2D';
  });
  draw(0);
})();
</script>
</div>"""


# ─── Map concept keywords → HTML builder functions ───────────────────────────
CONCEPT_VIZ_MAP = {
    'knn':             _html_knn,
    'k nearest':       _html_knn,
    'k-nearest':       _html_knn,
    'svm':             _html_svm,
    'support vector':  _html_svm,
    'decision tree':   _html_decision_tree,
    'neural network':  _html_neural_network,
    'gradient descent':_html_gradient_descent,
    'k-means':         _html_kmeans,
    'kmeans':          _html_kmeans,
    'k means':         _html_kmeans,
    'overfitting':     _html_overfitting,
    'underfitting':    _html_overfitting,
    'bias variance':   _html_overfitting,
    'pca':             _html_pca,
    'principal component': _html_pca,
    'random forest':   _html_random_forest,
    'linear regression': _html_linear_regression,
    'logistic regression': _html_logistic_regression,
    'backprop':        _html_backprop,
    'backpropagation': _html_backprop,
    'attention':       _html_attention,
    'transformer':     _html_attention,
    'regularization':  _html_regularization,
    'lasso':           _html_regularization,
    'ridge':           _html_regularization,
    'l1':              _html_regularization,
    'l2':              _html_regularization,
    'dbscan':          _html_dbscan,
    'cnn':             _html_cnn,
    'convolutional':   _html_cnn,
    'rnn':             _html_rnn,
    'lstm':            _html_rnn,
    'recurrent':       _html_rnn,
    'tsne':            _html_tsne,
    't-sne':           _html_tsne,
    't sne':           _html_tsne,
}


class VisualizationManager:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── Public API (matches existing rag_pipeline.py calls) ──────────────────

    def generate_visualization(self, query: str, context: str = "") -> Dict[str, Any]:
        """Main entry point — returns {success, html, code, error}"""
        print(f"🎨 Generating visualization for: {query[:60]}...")

        html = self._get_concept_html(query)

        if html:
            print("✓ Rich interactive visualization found")
            return {'success': True, 'html': html, 'code': '# Interactive HTML visualization', 'error': None}

        # Fallback: generate a Plotly chart via LLM
        print("↩ Falling back to LLM-generated Plotly chart")
        html, code = self._llm_plotly_fallback(query, context)
        if html:
            return {'success': True, 'html': html, 'code': code, 'error': None}

        return {'success': False, 'html': None, 'code': None, 'error': 'No visualization available'}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_concept_html(self, query: str) -> Optional[str]:
        q = query.lower()
        for keyword, builder in CONCEPT_VIZ_MAP.items():
            if keyword in q:
                try:
                    return builder()
                except Exception as e:
                    print(f"HTML builder error for '{keyword}': {e}")
                    traceback.print_exc()
        return None

    def _llm_plotly_fallback(self, query: str, context: str) -> tuple:
        """Generate a simple Plotly figure via Groq as last resort"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
            import numpy as np

            prompt = f"""Generate SHORT Python code using Plotly to visualize: {query}

RULES:
- Return ONLY Python code, no markdown
- Create variable called 'fig'
- Use plotly.express (px) only
- Max 20 lines
- No fig.show()
"""
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600
            )
            code = resp.choices[0].message.content.strip()
            code = re.sub(r'```python\s*', '', code)
            code = re.sub(r'```\s*', '', code).strip()

            local_vars = {'go': go, 'px': px, 'pd': pd, 'np': np, '__builtins__': __builtins__}
            exec(code, local_vars, local_vars)
            fig = local_vars.get('fig')
            if isinstance(fig, go.Figure):
                return fig.to_html(full_html=False, include_plotlyjs='cdn'), code
        except Exception as e:
            print(f"LLM fallback error: {e}")
        return None, None

    # ── Kept for backward compatibility ──────────────────────────────────────

    def should_visualize(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in CONCEPT_VIZ_MAP) or any(
            kw in q for kw in ['visualize','show','plot','diagram','explain how','how does','what is']
        )