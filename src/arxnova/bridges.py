"""Разметка мостов и калибровка шлюза (КО-1, КО-2, ФТ-8.4).

КО-1: 150–200 межстилевых пар-кандидатов, владелец слушает «конец A → начало B» и отмечает
«уместно / неуместно». Кандидаты выбираются по близости микса без порогов шлюза и равномерно по
её уровням — иначе разметка видела бы только то, что шлюз и так пропустит, и точность нечем было бы
проверить.
КО-2: перцентиль порогов подбирается под точность ≥ 90 %: из всех уровней, где среди принятых
шлюзом размеченных пар уместных не меньше 90 %, берётся самый мягкий (больше мостов)."""

from __future__ import annotations

import random
import sqlite3
import time

import numpy as np

TARGET = 180  # пар на разметку (КО-1: 150–200)
BINS = 6  # уровней близости микса
NEIGHBORS = 30  # из скольких ближайших соседей трека берутся кандидаты
PRECISION = 0.9  # КО-2
MIN_ACCEPTED = 5  # меньше принятых пар — точность не оценить


def ensure_candidates(conn: sqlite3.Connection, lib, seed: int = 7) -> int:
    """Набор кандидатов строится один раз и не меняется: разметка не должна «уплывать»."""
    have = conn.execute("SELECT count(*) FROM bridge_candidates").fetchone()[0]
    if have:
        return have
    styles = [lib.tracks[int(t)].style for t in lib.ids]
    sims = lib.vec @ lib.vec.T
    pairs: dict[tuple[int, int], float] = {}
    for i in range(len(lib.ids)):
        for j in np.argsort(-sims[i])[1:NEIGHBORS + 1]:
            if styles[i] != styles[j]:
                a, b = sorted((i, int(j)))
                pairs[(a, b)] = float(sims[i, j])
    if not pairs:
        return 0
    rng = random.Random(seed)
    vals = np.array(list(pairs.values()))
    edges = np.quantile(vals, np.linspace(0, 1, BINS + 1))
    chosen: list[tuple[int, int, float]] = []
    items = list(pairs.items())
    for k in range(BINS):
        lo, hi = edges[k], edges[k + 1]
        inbin = [(p, s) for p, s in items if lo <= s <= hi]
        rng.shuffle(inbin)
        chosen += [(a, b, s) for (a, b), s in inbin[: TARGET // BINS]]
    rng.shuffle(chosen)
    with conn:
        for order, (a, b, s) in enumerate(chosen):
            ta, tb = int(lib.ids[a]), int(lib.ids[b])
            if rng.random() < 0.5:  # кто играет первым — случайно
                ta, tb = tb, ta
            conn.execute("INSERT OR IGNORE INTO bridge_candidates (a, b, mix, ord) VALUES (?,?,?,?)", (ta, tb, s, order))
    return len(chosen)


def next_unlabeled(conn: sqlite3.Connection):
    return conn.execute(
        """SELECT c.a, c.b, c.mix FROM bridge_candidates c
           LEFT JOIN bridge_labels l ON l.a = c.a AND l.b = c.b
           WHERE l.a IS NULL ORDER BY c.ord LIMIT 1""").fetchone()


def progress(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT count(*) FROM bridge_candidates").fetchone()[0]
    done = conn.execute(
        "SELECT label, count(*) FROM bridge_labels l JOIN bridge_candidates c ON c.a = l.a AND c.b = l.b GROUP BY label"
    ).fetchall()
    d = {r[0]: r[1] for r in done}
    return {"total": total, "good": d.get("good", 0), "bad": d.get("bad", 0), "skip": d.get("skip", 0)}


def label(conn: sqlite3.Connection, a: int, b: int, value: str) -> None:
    if value not in ("good", "bad", "skip"):
        raise ValueError(value)
    with conn:
        conn.execute("INSERT OR REPLACE INTO bridge_labels (a, b, label, ts) VALUES (?,?,?,?)",
                     (a, b, value, int(time.time())))


def calibrate(conn: sqlite3.Connection, lib) -> dict:
    """КО-2: точность шлюза на размеченных парах для каждого перцентиля порогов."""
    labeled = [(a, b, lab) for a, b, lab in conn.execute(
        "SELECT a, b, label FROM bridge_labels WHERE label IN ('good', 'bad')") if a in lib.row and b in lib.row]
    rows = []
    best = None
    for pct in range(75, 100):
        acc = [lab for a, b, lab in labeled if lib.gate.check(lib.row[a], lib.row[b], pct=float(pct)).ok]
        good = sum(1 for x in acc if x == "good")
        prec = good / len(acc) if acc else None
        rows.append({"pct": pct, "accepted": len(acc), "good": good, "precision": prec})
        if best is None and prec is not None and len(acc) >= MIN_ACCEPTED and prec >= PRECISION:
            best = pct
    goods = sum(1 for *_, lab in labeled if lab == "good")
    return {"labeled": len(labeled), "good": goods, "levels": rows, "suggested_pct": best}


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arx Nova — мосты</title>
<style>
:root{--bg:#0B0C0E;--surface:#141518;--line:#26272B;--text:#F2EFE9;--sec:#98938A;--muted:#6F6B62;--brass:#C9A86A;--on:#1A1509}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--sec);margin:0 0 22px;font-size:14px}
.bar{height:4px;background:var(--line);border-radius:2px;margin:0 0 24px}.bar>i{display:block;height:100%;background:var(--brass);border-radius:2px;width:0}
.pair{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px;min-height:96px}
.card.on{border-color:var(--brass)}.style{color:var(--brass);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.t{font-weight:600;margin-top:4px}.a{color:var(--sec);font-size:14px}.arrow{color:var(--muted);font-size:22px}
.now{color:var(--sec);font-size:13px;margin:14px 0 18px;min-height:18px}
.btns{display:flex;gap:10px;flex-wrap:wrap}
button{font:inherit;border-radius:22px;padding:10px 18px;border:1px solid var(--line);background:var(--surface);color:var(--text);cursor:pointer}
button.good{background:var(--brass);border-color:var(--brass);color:var(--on);font-weight:600}
button:disabled{opacity:.4;cursor:default}
.keys{color:var(--muted);font-size:12px;margin-top:10px}
section{margin-top:40px;border-top:1px solid var(--line);padding-top:20px}
table{border-collapse:collapse;width:100%;font-size:13px;color:var(--sec)}td,th{padding:4px 6px;text-align:right}th{color:var(--muted);font-weight:500}
tr.best td{color:var(--brass)}
@media (max-width:560px){.pair{grid-template-columns:1fr}.arrow{display:none}}
</style></head><body><main>
<h1>Мосты между стилями</h1>
<p class="sub">Слушайте, как конец первого трека переходит в начало второго, и отметьте, уместен ли такой переход в потоке.
По этим оценкам шлюз выберет пороги: лучше меньше мостов, но все уместные.</p>
<nav style="margin:-8px 0 16px"><a href="/ui/bridges" style="color:var(--brass);margin-right:16px;text-decoration:none">Разметка мостов</a><a href="/ui/gate" style="color:var(--sec);text-decoration:none">Шлюз</a></nav>
<div class="bar"><i id="bar"></i></div>
<div id="work">
 <div class="pair"><div class="card" id="ca"></div><div class="arrow">→</div><div class="card" id="cb"></div></div>
 <div class="now" id="now"></div>
 <div class="btns">
  <button class="good" id="g">Уместно</button><button id="bd">Неуместно</button>
  <button id="sk">Пропустить</button><button id="rp">Переслушать</button>
 </div>
 <div class="keys">Клавиши: 1 — уместно, 2 — неуместно, 3 — пропустить, пробел — переслушать</div>
</div>
<div id="fin" hidden><p>Все пары размечены. Спасибо!</p></div>
<section><h2 style="font-size:17px;margin:0 0 6px">Калибровка шлюза</h2>
<p class="sub" id="cal">Появится после разметки.</p><div id="tbl"></div>
<button class="good" id="apply" hidden>Применить</button></section>
</main>
<audio id="au" preload="auto"></audio>
<script>
const $=id=>document.getElementById(id), au=$('au');
let cur=null, stopAt=0, phase=0, best=null;
const SEG=20;
function card(el,t){el.innerHTML='<div class="style">'+esc(t.style)+'</div><div class="t">'+esc(t.title||'—')+'</div><div class="a">'+esc(t.artist||'')+'</div>'}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c])}
function play(t,from,to,label){au.src='/v1/local/audio/'+t.id;au.onloadedmetadata=()=>{au.currentTime=t.offset+from;au.play()};stopAt=t.offset+to;$('now').textContent=label}
function start(){if(!cur)return;phase=1;$('ca').classList.add('on');$('cb').classList.remove('on');
 const a=cur.a, da=a.duration||60; play(a,Math.max(0,da-SEG-4),da-4,'Играет конец первого трека…')}
au.ontimeupdate=()=>{if(au.currentTime<stopAt)return;au.pause();
 if(phase===1){phase=2;$('ca').classList.remove('on');$('cb').classList.add('on');play(cur.b,2,2+SEG,'…и начало второго')}
 else if(phase===2){phase=0;$('cb').classList.remove('on');$('now').textContent='Уместен ли переход?'}};
async function next(){const r=await (await fetch('/v1/local/bridges/next')).json();
 const n=r.good+r.bad+r.skip;$('bar').style.width=(r.total?100*n/r.total:0)+'%';
 if(r.done){$('work').hidden=true;$('fin').hidden=false;cur=null;au.pause();calib();return}
 cur=r;card($('ca'),r.a);card($('cb'),r.b);$('now').textContent='Размечено '+n+' из '+r.total;start();if(n&&n%10===0)calib()}
async function mark(l){if(!cur)return;au.pause();await fetch('/v1/local/bridges/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({a:cur.a.id,b:cur.b.id,label:l})});next()}
async function calib(){const r=await (await fetch('/v1/local/bridges/calibration')).json();best=r.suggested_pct;
 if(!r.facets){$('cal').textContent='Звук медиатеки ещё разбирается (arxnova facets) — шлюз пока закрыт, калибровка позже.';return}
 if(r.labeled<20){$('cal').textContent='Размечено '+r.labeled+' — калибровка от 20 пар.';return}
 $('cal').textContent=best?('Точность ≥ 90 % с перцентилем '+best+' (сейчас '+r.current_pct+').'):'Пока ни один уровень не даёт 90 % точности — разметьте ещё.';
 $('tbl').innerHTML='<table><tr><th>перцентиль</th><th>пропущено шлюзом</th><th>уместных</th><th>точность</th></tr>'+
  r.levels.filter(x=>x.accepted).map(x=>'<tr'+(x.pct===best?' class="best"':'')+'><td>'+x.pct+'</td><td>'+x.accepted+'</td><td>'+x.good+'</td><td>'+Math.round(100*x.precision)+' %</td></tr>').join('')+'</table>';
 $('apply').hidden=!best}
$('g').onclick=()=>mark('good');$('bd').onclick=()=>mark('bad');$('sk').onclick=()=>mark('skip');$('rp').onclick=start;
$('apply').onclick=async()=>{await fetch('/v1/local/bridges/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pct:best})});calib()};
document.onkeydown=e=>{if(e.key==='1')mark('good');else if(e.key==='2')mark('bad');else if(e.key==='3')mark('skip');else if(e.key===' '){e.preventDefault();start()}};
next();calib();
</script></body></html>
"""


GATE_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arx Nova — шлюз</title>
<style>
:root{--bg:#0B0C0E;--surface:#141518;--line:#26272B;--text:#F2EFE9;--sec:#98938A;--muted:#6F6B62;--brass:#C9A86A;--on:#1A1509}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 60px}
nav a{color:var(--sec);margin-right:16px;text-decoration:none}nav a.on{color:var(--brass)}
h1{font-size:22px;margin:14px 0 4px}h2{font-size:16px;margin:0 0 10px}.sub{color:var(--sec);font-size:14px;margin:0 0 18px}
section{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px;margin:14px 0}
input[type=range]{width:100%;accent-color:var(--brass)}.ends{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}
select,button{font:inherit;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:18px;padding:6px 12px}
button.p{background:var(--brass);border-color:var(--brass);color:var(--on);font-weight:600;cursor:pointer}
.chip{display:inline-flex;gap:8px;align-items:center;border:1px solid var(--line);border-radius:16px;padding:4px 10px;margin:4px 6px 0 0;font-size:13px}
.chip b{cursor:pointer;color:var(--muted);font-weight:400}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:4px 0;border-bottom:1px solid var(--line)}td.n{text-align:right;color:var(--sec)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.ok{color:var(--brass)}.no{color:var(--muted)}
.log div{padding:5px 0;border-bottom:1px solid var(--line);font-size:13px}.log small{color:var(--muted)}
canvas{width:100%;height:440px;display:block;border-radius:10px;background:#08090A}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;display:none}
.legend span{display:inline-block;margin:6px 10px 0 0;font-size:12px;color:var(--sec)}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
</style></head><body><main>
<nav><a href="/ui/bridges">Разметка мостов</a><a class="on" href="/ui/gate">Шлюз</a></nav>
<h1>Межжанровый шлюз</h1>
<p class="sub" id="cover"></p>
<section><h2>Смелость подмешивания</h2>
<input type="range" min="0" max="1" step="0.05" id="bold"><div class="ends"><span>Только самые уверенные мосты</span><span id="bv"></span><span>Смелее</span></div></section>
<section><h2>Не переходить между стилями</h2>
<div class="row"><select id="s1"></select><span>и</span><select id="s2"></select><button class="p" id="add">Запретить</button></div>
<div id="pairs"></div></section>
<section><h2>Решения шлюза за 7 дней</h2><table id="counts"></table></section>
<section><h2>Последние переходы</h2><div class="log" id="log"></div></section>
<section><h2>Карта медиатеки</h2><p class="sub">Похожие по звучанию треки рядом, цвет — стиль. Мосты — на стыках облаков.</p>
<canvas id="map"></canvas><div class="legend" id="legend"></div></section>
</main><div id="tip"></div>
<script>
const $=id=>document.getElementById(id);let S=null;
const COND={accepted:'Пропущено — мосты',mix:'Далеко по общему звучанию',facet:'Разошлись по ритму, звучанию или вокалу',tempo:'Несовместимый темп',
 energy:'Разная энергия',vocal:'У одного есть вокал, у другого нет',confidence:'Стиль трека определён неуверенно',mutual:'Не взаимные соседи',
 pair:'Пара стилей запрещена',no_facets:'Звук трека ещё не разобран'};
const FAC={rhythm:'ритм',instr:'звучание',vocal:'вокал'};
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c])}
async function load(){S=await (await fetch('/v1/local/gate/state')).json();
 $('cover').textContent='Звук разобран у '+S.facets+' из '+S.tracks+' треков. Порог калибровки — '+S.gate_pct+'-й перцентиль внутристилевых соседей.';
 $('bold').value=S.boldness;bv();
 const opts=S.styles.map(s=>'<option>'+esc(s)+'</option>').join('');$('s1').innerHTML=opts;$('s2').innerHTML=opts;
 $('pairs').innerHTML=S.forbid.map((p,i)=>'<span class="chip">'+esc(p[0])+' ↔ '+esc(p[1])+' <b data-i="'+i+'">✕</b></span>').join('')||'<p class="sub" style="margin:8px 0 0">Запретов нет.</p>';
 document.querySelectorAll('#pairs b').forEach(b=>b.onclick=()=>{S.forbid.splice(+b.dataset.i,1);save({forbid:S.forbid})});
 const c=Object.entries(S.counts).sort((a,b)=>b[1]-a[1]);
 $('counts').innerHTML=c.length?c.map(([k,n])=>'<tr><td>'+esc(COND[k]||k)+'</td><td class="n">'+n+'</td></tr>').join(''):'<tr><td class="no">Переходов в другой стиль ещё не было.</td></tr>';
 $('log').innerHTML=S.recent.map(r=>'<div><span class="'+(r.accepted?'ok':'no')+'">'+(r.accepted?'✓':'✕')+'</span> '+esc(r.from.title)+' <small>'+esc(r.from.style)+
  '</small> → '+esc(r.to.title)+' <small>'+esc(r.to.style)+'</small> <small>'+(r.accepted?'':esc(COND[r.failed]||r.failed)+(r.detail.facet?' ('+esc(FAC[r.detail.facet]||r.detail.facet)+')':''))+
  '</small></div>').join('')||'<p class="sub" style="margin:0">Пусто.</p>';}
function bv(){const b=+$('bold').value;$('bv').textContent=Math.round(b*100)+' %'}
async function save(x){await fetch('/v1/local/gate/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(x)});load()}
$('bold').oninput=bv;$('bold').onchange=()=>save({boldness:+$('bold').value});
$('add').onclick=()=>{const a=$('s1').value,b=$('s2').value;if(a&&b&&a!==b){S.forbid.push([a,b]);save({forbid:S.forbid})}};
let HUES={};function hue(s){return HUES[s]??0}
async function map(){const r=await (await fetch('/v1/local/map')).json(),P=r.points,cv=$('map');
 if(!P.length){cv.replaceWith(Object.assign(document.createElement('p'),{className:'sub',textContent:'Карта появится после arxnova facets.'}));return}
 const dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);
 const px=p=>[14+p[0]*(W-28),14+(1-p[1])*(H-28)];
 // крупные стили — заметно разными оттенками (золотой угол), а не похожими по хешу
 const byN={};P.forEach(p=>byN[p[2]]=(byN[p[2]]||0)+1);
 Object.entries(byN).sort((a,b)=>b[1]-a[1]).forEach(([s],i)=>HUES[s]=(40+i*137.5)%360);
 for(const p of P){const [x,y]=px(p);g.fillStyle='hsl('+hue(p[2])+',55%,62%)';g.beginPath();g.arc(x,y,3.2,0,7);g.fill()}
 const cnt={};P.forEach(p=>cnt[p[2]]=(cnt[p[2]]||0)+1);
 $('legend').innerHTML=Object.entries(cnt).sort((a,b)=>b[1]-a[1]).slice(0,12).map(([s,n])=>'<span><i style="background:hsl('+hue(s)+',55%,62%)"></i>'+esc(s)+' · '+n+'</span>').join('');
 cv.onmousemove=e=>{const b=cv.getBoundingClientRect(),mx=e.clientX-b.left,my=e.clientY-b.top;let best=null,d=144;
  for(const p of P){const [x,y]=px(p),q=(x-mx)**2+(y-my)**2;if(q<d){d=q;best=p}}
  const t=$('tip');if(!best){t.style.display='none';return}
  t.innerHTML=esc(best[3])+'<br><span style="color:var(--sec)">'+esc(best[4])+' · '+esc(best[2])+'</span>';t.style.left=(e.clientX+12)+'px';t.style.top=(e.clientY+12)+'px';t.style.display='block'};
 cv.onmouseleave=()=>$('tip').style.display='none'}
load();map();
</script></body></html>
"""
