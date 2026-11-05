"""Табло подбора: как поток заходит на самом деле (решение 0013, раздел «как мерить»). Считает только показы потока, строго:
дослушал — от 95 %. Числа разделены по прошлому исходу: общее среднее зависит от того, сколько было цепочек пропусков."""

from __future__ import annotations

import json
import sqlite3
import time

GAP_MS = 45 * 60_000
STEPS = ("ранний", "средний", "поздний", "дослушал", "сильный")


def _step(et: str, pct: int, strong: bool) -> int:
    if et == "replay" or strong:
        return 4
    if et == "listen":
        return 3
    return 0 if pct < 15 else 1 if pct < 60 else 2


def _why_of(result: str | None, track: int) -> str:
    for p in json.loads(result or "[]"):
        if p.get("track") == track:
            if p.get("why"):
                return p["why"]
            text = p.get("reason", "")
            return "bridge" if "мост" in text else "spice" if "подмес" in text else "restyle" if "предложить" in text else "обычный"
    return "обычный"


def report(conn: sqlite3.Connection, device_ids: list[int], days: int = 30, now_ms: int | None = None) -> dict:
    now_ms = now_ms or int(time.time() * 1000)
    since = now_ms - days * 86_400_000
    q = ",".join("?" * len(device_ids))
    rows = conn.execute(
        f"""SELECT e.ts, e.track_id, e.event_type, e.played_ratio_pct, e.track_length_s, e.is_organic, e.nova_request_id, e.left_by, r.result
            FROM events e LEFT JOIN nova_requests r ON r.id = e.nova_request_id
            WHERE e.device_id IN ({q}) AND e.ts >= ? AND e.track_id IS NOT NULL
              AND e.event_type IN ('listen', 'replay', 'skip', 'like', 'similar') ORDER BY e.ts""",
        (*device_ids, since)).fetchall()
    strong_at = {}
    for r in rows:
        if r[2] in ("like", "similar"):
            strong_at.setdefault(r[1], []).append(r[0])
    shows, last_ts, session = [], None, 0
    last_show: dict[int, dict] = {}
    for ts, tid, et, pct, length, organic, _req, left_by, result in rows:
        if et not in ("listen", "replay", "skip"):
            continue
        if last_ts is not None and ts - last_ts > GAP_MS:
            session += 1
        last_ts = ts
        span = max((length or 0) * 1000, 60_000)
        strong = any(ts - span <= x <= ts + 90_000 for x in strong_at.get(tid, ()))
        earlier = last_show.get(tid)
        if et == "replay" and earlier is not None and earlier["session"] == session:
            earlier["step"] = 4  # вернулся и дослушал: это исход того показа. Плеер пишет возврат как свой выбор, и в попадания он не попадал
            continue
        show = {"session": session, "step": _step(et, pct or 0, strong), "organic": bool(organic), "left_by": left_by,
                "why": _why_of(result, tid)}
        shows.append(show)
        last_show[tid] = show
    by_prev: dict[str, list[int]] = {}
    by_why: dict[str, list[int]] = {}
    chains = {"всего": 0, "поток угадал": 0, "сам выбрал": 0, "бросил": 0, "пропусков": 0}
    calm = 0
    for i, s in enumerate(shows):
        before = shows[i - 1] if i and shows[i - 1]["session"] == s["session"] else None
        refused = before is not None and before["step"] <= 1 and before["left_by"] != "back"
        if not s["organic"]:
            kind = "первый в сессии" if before is None else "после пропуска" if refused else "после дослушанного"
            by_prev.setdefault(kind, [0] * 5)[s["step"]] += 1
            by_why.setdefault(s["why"] or "обычный", [0] * 5)[s["step"]] += 1
            calm += kind == "после дослушанного"
        if s["step"] <= 1 and s["left_by"] != "back" and not s["organic"] and not refused:
            chains["всего"] += 1
            j = i
            while j + 1 < len(shows) and shows[j + 1]["session"] == s["session"] and shows[j + 1]["step"] <= 1:
                j += 1
            chains["пропусков"] += j - i + 1
            after = shows[j + 1] if j + 1 < len(shows) and shows[j + 1]["session"] == s["session"] else None
            chains["бросил" if after is None else "сам выбрал" if after["organic"] else "поток угадал"] += 1
    total = [sum(v[k] for v in by_prev.values()) for k in range(5)]
    n = sum(total)
    return {"days": days, "shows": n, "total": total, "by_prev": by_prev, "by_why": by_why, "chains": chains,
            "misses_per_100": round(100 * total[0] / n, 1) if n else None,
            "hits_per_100": round(100 * total[4] / n, 1) if n else None,
            "done_share": round((total[3] + total[4]) / n, 3) if n else None,
            "chain_starts_per_100_calm": round(100 * chains["всего"] / calm, 1) if calm else None}


def format_report(r: dict) -> str:
    def line(name: str, x: list[int]) -> str:
        n = sum(x)
        return f"  {name:22s} {n:5d}  " + "  ".join(f"{100 * v / n:5.1f}" for v in x) if n else f"  {name:22s}     0"

    head = f"  {'':22s} {'показов':>5s}  " + "  ".join(f"{s[:5]:>5s}" for s in STEPS)
    out = [f"Поток за {r['days']} дней: показов {r['shows']}", "",
           f"  промахи на 100 показов    {r['misses_per_100']}", f"  попадания на 100 показов  {r['hits_per_100']}",
           f"  дослушано (от 95 %)       {r['done_share']}", f"  цепочек пропусков на 100 спокойных показов  {r['chain_starts_per_100_calm']}", "",
           "Что было перед треком, % показов:", head]
    out += [line(k, v) for k, v in sorted(r["by_prev"].items())]
    out += ["", "По причине выбора, % показов:", head]
    out += [line(k, v) for k, v in sorted(r["by_why"].items())]
    c = r["chains"]
    out += ["", f"Цепочки пропусков: {c['всего']}, пропусков в них {c['пропусков']}; поток угадал {c['поток угадал']}, сам выбрал {c['сам выбрал']}, бросил слушать {c['бросил']}"]
    return "\n".join(out)
