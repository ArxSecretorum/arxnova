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


def _show_rows(conn: sqlite3.Connection, device_ids: list[int], since: int) -> list:
    q = ",".join("?" * len(device_ids))
    return conn.execute(
        f"""SELECT e.ts, e.track_id, e.event_type, e.played_ratio_pct, e.track_length_s, e.is_organic, e.nova_request_id, e.left_by, r.result
            FROM events e LEFT JOIN nova_requests r ON r.id = e.nova_request_id
            WHERE e.device_id IN ({q}) AND e.ts >= ? AND e.track_id IS NOT NULL
              AND e.event_type IN ('listen', 'replay', 'skip', 'like', 'similar') ORDER BY e.ts""",
        (*device_ids, since)).fetchall()


def _shows(rows: list) -> list[dict]:
    """Показы по событиям: исход трека, сессия по разрыву, причина выбора, запрос, из которого трек пришёл.
    Плеер шлёт лайк без запроса, а «Похожие» и возврат — как свой выбор и тоже без запроса: они не показы, а исход
    соседнего показа того же трека (лайк и «Похожие» рядом по времени, возврат — в той же сессии)."""
    strong_at = {}
    for r in rows:
        if r[2] in ("like", "similar"):
            strong_at.setdefault(r[1], []).append(r[0])
    shows, last_ts, session = [], None, 0
    last_show: dict[int, dict] = {}
    for ts, tid, et, pct, length, organic, req, left_by, result in rows:
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
                "why": _why_of(result, tid), "track": tid, "req": req, "ts": ts}
        shows.append(show)
        last_show[tid] = show
    return shows


def report(conn: sqlite3.Connection, device_ids: list[int], days: int = 30, now_ms: int | None = None) -> dict:
    now_ms = now_ms or int(time.time() * 1000)
    shows = _shows(_show_rows(conn, device_ids, now_ms - days * 86_400_000))
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


TRIAL_MIN = 20  # меньше расхождений в рукаве — разность не считается: шум


def trial_report(conn: sqlite3.Connection, device_ids: list[int], days: int = 180, now_ms: int | None = None,
                 boot: int = 2000) -> dict:
    """Чередование таблицы «кто слушает вместе» (решение 0015). Показ — как в табло (`_shows`): лайк, «Похожие» и возврат
    засчитываются своему показу, хотя плеер шлёт их без запроса. Входит показ трека из основного подбора запроса с рукавом (не
    проба и не мост), сыгранный не по выбору слушателя. «Не предлагать» — ранний пропуск: после него плеер пропуска не пишет.
    «Разошлись» — другой рукав на этом месте пачки выбрал бы другой трек: сравниваются только такие показы."""
    import bisect
    import random

    now_ms = now_ms or int(time.time() * 1000)
    since = now_ms - days * 86_400_000
    q = ",".join("?" * len(device_ids))
    reqs = {}
    for rid, detail in conn.execute(
            f"SELECT id, detail FROM nova_requests WHERE device_id IN ({q}) AND ts >= ? AND detail LIKE '%\"arm\"%'", (*device_ids, since)):
        d = (json.loads(detail) or {}).get("listen") or {}
        if d.get("arm") in ("A", "B"):
            reqs[rid] = d
    if not reqs:
        return {"requests": 0, "shows": 0}
    shows = _shows(_show_rows(conn, device_ids, since))
    outcome: dict[tuple[int, int], dict] = {}
    for s in shows:
        key = (s["req"], s["track"])
        if s["req"] in reqs and not s["organic"] and key not in outcome:
            outcome[key] = {"step": s["step"], "session": s["session"]}
    stamps = [s["ts"] for s in shows]
    for ts, tid, rid in conn.execute(
            f"""SELECT ts, track_id, nova_request_id FROM events WHERE device_id IN ({q}) AND ts >= ? AND event_type = 'dislike'
                  AND nova_request_id IS NOT NULL AND track_id IS NOT NULL""", (*device_ids, since)):
        if rid not in reqs:
            continue
        if (rid, tid) in outcome:
            outcome[(rid, tid)]["step"] = 0
            continue
        i = bisect.bisect_right(stamps, ts) - 1  # сессия «Не предлагать» без показа — по ближайшему показу перед ним
        session = shows[i]["session"] if i >= 0 and ts - stamps[i] <= GAP_MS else -1 - len(outcome)
        outcome[(rid, tid)] = {"step": 0, "session": session}
    out = []
    for (rid, tid), o in outcome.items():
        d = reqs[rid]
        main, counter = d.get("main") or [], d.get("counter") or []
        if tid not in main or main.index(tid) >= len(counter):
            continue
        out.append({"arm": d["arm"], "differ": counter[main.index(tid)] != tid, "step": o["step"], "session": o["session"]})
    res: dict = {"requests": len(reqs), "shows": len(out)}
    for arm in "AB":
        sub = [x for x in out if x["arm"] == arm and x["differ"]]
        n = len(sub)
        res[arm] = {"differ": n, "same": sum(1 for x in out if x["arm"] == arm and not x["differ"]),
                    "done": round(sum(x["step"] >= 3 for x in sub) / n, 3) if n else None,
                    "early": round(sum(x["step"] == 0 for x in sub) / n, 3) if n else None}
    if res["A"]["differ"] >= TRIAL_MIN and res["B"]["differ"] >= TRIAL_MIN:
        diff = [x for x in out if x["differ"]]
        sessions = sorted({x["session"] for x in diff})
        by = {s: [x for x in diff if x["session"] == s] for s in sessions}
        rng = random.Random(7)
        for name, hit in (("done", lambda x: x["step"] >= 3), ("early", lambda x: x["step"] == 0)):
            draws = []
            for _ in range(boot):
                xs = [x for s in rng.choices(sessions, k=len(sessions)) for x in by[s]]
                a = [hit(x) for x in xs if x["arm"] == "A"]
                b = [hit(x) for x in xs if x["arm"] == "B"]
                if a and b:
                    draws.append(sum(b) / len(b) - sum(a) / len(a))
            draws.sort()
            point = res["B"][name] - res["A"][name]
            res[name + "_diff"] = (round(point, 3), round(draws[int(0.025 * len(draws))], 3), round(draws[int(0.975 * len(draws)) - 1], 3))
    return res


def format_trial(r: dict) -> str:
    if not r.get("requests"):
        return "Чередование таблицы слушателей: запросов с рукавом ещё нет"
    out = [f"Чередование таблицы слушателей (решение 0015): запросов {r['requests']}, показов из основного подбора {r['shows']}",
           "  рукав           выбор разошёлся  дослушано  ранних пропусков  совпал с другим рукавом"]
    for arm, name in (("A", "A — как раньше"), ("B", "B — с таблицей")):
        x = r[arm]
        pct = lambda v: f"{100 * v:5.1f} %" if v is not None else "    —  "
        out.append(f"  {name:15s} {x['differ']:15d}  {pct(x['done']):>9s}  {pct(x['early']):>16s}  {x['same']:23d}")
    if "done_diff" in r:
        d, e = r["done_diff"], r["early_diff"]
        out.append(f"  B − A: дослушано {100 * d[0]:+.1f} [{100 * d[1]:+.1f}…{100 * d[2]:+.1f}] п. п., "
                   f"ранних пропусков {100 * e[0]:+.1f} [{100 * e[1]:+.1f}…{100 * e[2]:+.1f}] п. п. (95 %, бутстрэп по сессиям)")
    else:
        out.append(f"  разность считается от {TRIAL_MIN} разошедшихся показов в каждом рукаве")
    return "\n".join(out)
