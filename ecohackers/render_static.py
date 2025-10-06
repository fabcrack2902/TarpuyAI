# render_static.py
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
STATIC = BASE / "static"
SRC = STATIC / "index.html"
OUT = STATIC / "output.html"

if not SRC.exists():
    raise SystemExit(f"Template not found: {SRC}")

CLIMATE_JS = r'''
<script>
/* Datos climáticos embebidos */
const CLIMATE_DATA_RAW = `Fecha,Var1,Precipitacion1,Var2,Precipitacion2,Temp_C,Humedad_pct,Var3,Optimo
1/12/2024,35498.59,0,270.73,0,25.24,12.09,520.58,Sí
2/12/2024,35449.37,0,282.32,0,24.26,9.94,446.92,Sí
3/12/2024,35402.66,0,253.44,0,21.45,9.46,431.97,Sí
4/12/2024,35358.49,0,243.87,0,20.02,8.73,445.17,Sí
5/12/2024,35316.88,0,266.36,0,20.82,7.92,386.31,No
6/12/2024,35277.88,0,306.83,0,27.74,9.72,440.02,Sí
7/12/2024,35241.49,0,286.91,0,27.41,11.86,512.49,Sí
8/12/2024,35207.75,0,265.72,0,22.52,9.3,427.08,Sí
9/12/2024,35176.68,0,254.64,0,21.1,8.8,415.49,Sí
10/12/2024,35148.3,0,270.27,0,23.09,9.52,433.68,Sí
11/12/2024,35122.63,0,238.28,0,21.46,10.26,497.6,Sí
12/12/2024,35099.68,0,195.04,0,16.85,8.26,594.49,No
13/12/2024,35079.48,0,212.71,0,18.35,8.92,553.63,Sí
14/12/2024,35062.04,0,237.94,0,19.52,8.64,466.55,Sí
15/12/2024,35047.36,0,254.46,0,20.22,8.12,407.22,Sí
16/12/2024,35035.47,0,279.38,0,23.06,8.71,409.05,Sí`;

/* Parse CSV y añadir datasets al tsChart (preserva NDVI dataset que ya existe) */
(function(){
  function parse(csv){
    const lines = csv.trim().split('\n');
    const headers = lines.shift().split(',').map(h=>h.trim());
    return lines.map(l=>{
      const cols = l.split(',').map(c=>c.trim());
      const obj = {};
      headers.forEach((h,i)=>obj[h]=cols[i]??'');
      const parts = obj.Fecha.includes('/') ? obj.Fecha.split('/') : obj.Fecha.split('-');
      obj._date = (parts.length===3 && obj.Fecha.includes('/')) ? new Date(parts[2], Number(parts[1])-1, parts[0]) : new Date(obj.Fecha);
      obj.Temp_C = parseFloat((obj['Temp (°C)']||obj.Temp_C||'').toString().replace(',', '.'))||null;
      obj.Humedad_pct = parseFloat((obj['Humedad (%)']||obj.Humedad_pct||'').toString().replace(',', '.'))||null;
      obj.Var3 = parseFloat((obj.Var3||'').toString().replace(',', '.'))||null;
      obj.Optimo = String(obj['Óptimo para flores']||obj.Optimo||'').toLowerCase().startsWith('s');
      return obj;
    });
  }

  if (typeof tsChart === 'undefined') return;
  const data = parse(CLIMATE_DATA_RAW);
  const labels = data.map(d=>d._date.toISOString().slice(0,10));
  const temps = data.map(d=>d.Temp_C);
  const hums = data.map(d=>d.Humedad_pct);
  const var3 = data.map(d=>d.Var3);
  const optPoints = data.map((d,i)=> d.Optimo ? {x: labels[i], y: temps[i]} : null).filter(Boolean);

  const climateDatasets = [
    { label: 'Temp (°C)', data: temps, borderColor: 'tomato', backgroundColor: 'rgba(255,99,71,0.12)', fill:false, hidden:true },
    { label: 'Humedad (%)', data: hums, borderColor: 'skyblue', backgroundColor: 'rgba(135,206,235,0.12)', fill:false, hidden:true },
    { label: 'Var3', data: var3, borderColor: 'green', backgroundColor: 'rgba(0,128,0,0.08)', fill:false, hidden:true },
    { label: 'Días óptimos', data: optPoints, type:'scatter', backgroundColor:'gold', pointRadius:6, showLine:false, hidden:true }
  ];

  tsChart.data.datasets = tsChart.data.datasets.slice(0,1).concat(climateDatasets);
  tsChart.data.labels = labels;
  tsChart.update();

  // toggle button
  const chartsDiv = document.getElementById('charts') || document.body;
  const btn = document.createElement('button');
  btn.textContent = 'Mostrar clima';
  btn.style.marginTop='6px';
  btn.style.padding='6px 8px';
  btn.style.fontSize='12px';
  let shown=false;
  btn.addEventListener('click', ()=>{
    shown = !shown;
    for (let i=1;i<tsChart.data.datasets.length;i++) tsChart.data.datasets[i].hidden = !shown;
    tsChart.update();
    btn.textContent = shown ? 'Ocultar clima' : 'Mostrar clima';
  });
  chartsDiv.appendChild(btn);

  // compact table
  const wrap = document.createElement('div');
  wrap.style.marginTop='8px';
  wrap.style.maxHeight='140px';
  wrap.style.overflowY='auto';
  wrap.style.fontSize='12px';
  const rows = data.map(d=>`<tr><td style="padding:4px 6px">${d._date.toISOString().slice(0,10)}</td><td style="padding:4px 6px;text-align:right">${d.Temp_C??''}</td><td style="padding:4px 6px;text-align:right">${d.Humedad_pct??''}</td><td style="padding:4px 6px;text-align:center">${d.Optimo?'Sí':'No'}</td></tr>`).join('');
  wrap.innerHTML = `<table style="width:100%;border-collapse:collapse"><thead><tr style="background:#f6f6f6;font-weight:600"><th style="text-align:left;padding:6px">Fecha</th><th style="text-align:right;padding:6px">Temp</th><th style="text-align:right;padding:6px">Humedad</th><th style="text-align:center;padding:6px">Óptimo</th></tr></thead><tbody>${rows}</tbody></table>`;
  chartsDiv.appendChild(wrap);
})();
</script>
'''

# read template
html = SRC.read_text(encoding="utf-8")

# inject CLIMATE_JS before closing body
if "</body>" in html:
    out_html = html.replace("</body>", CLIMATE_JS + "\n</body>")
else:
    out_html = html + CLIMATE_JS

OUT.write_text(out_html, encoding="utf-8")
print(f"Generated {OUT}")
