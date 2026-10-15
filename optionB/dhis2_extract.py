"""Extraction from the SNIS DHIS2 instance (snisrdc.com) for the zone-level study.

Credentials are read from the environment and never written to disk:

    export DHIS2_URL=https://snisrdc.com          # base URL, no trailing /api
    export DHIS2_USER=...
    export DHIS2_PASS=...

Steps (run in this order; each writes to optionB/data/<step>/):

    python dhis2_extract.py system                 # server version, last analytics run
    python dhis2_extract.py find-elements chol     # candidate data elements (resolves question 1)
    python dhis2_extract.py element <DE_UID>       # metadata: created date, datasets (questions 2-3)
    python dhis2_extract.py orgunits <LEVEL>       # health-zone list and hierarchy
    python dhis2_extract.py analytics <DE_UID> <DS_UID> <LEVEL> 2020 2026
                                                   # cases + completeness/timeliness per zone-week
    python dhis2_extract.py snapshot <DE_UID> <DS_UID>
                                                   # raw data values with lastUpdated (vintages)

Endpoint names follow the standard DHIS2 Web API. Check them against the
server version printed by `system` before a full pull.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "data"
METRICS = ("REPORTING_RATE", "ACTUAL_REPORTS", "EXPECTED_REPORTS",
           "REPORTING_RATE_ON_TIME", "ACTUAL_REPORTS_ON_TIME")


def session():
    try:
        base = os.environ["DHIS2_URL"].rstrip("/")
        auth = (os.environ["DHIS2_USER"], os.environ["DHIS2_PASS"])
    except KeyError as e:
        sys.exit(f"missing environment variable {e}")
    s = requests.Session()
    s.auth = auth
    s.headers["Accept"] = "application/json"
    return s, base + "/api"


def get(s, url, params=None, retries=3):
    for k in range(retries):
        r = s.get(url, params=params, timeout=300)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (409, 500, 502, 503, 504) and k < retries - 1:
            time.sleep(10 * (k + 1))
            continue
        raise RuntimeError(f"{r.status_code} {r.url}\n{r.text[:500]}")


def save(step, name, obj):
    d = OUT / step
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if isinstance(obj, (dict, list)):
        p.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    else:
        p.write_text(obj)
    print("wrote", p)
    return p


def iso_weeks(y0, y1):
    """All ISO weeks y0-W1 .. y1-Wlast, in DHIS2 period syntax (2022W1)."""
    out = []
    for y in range(y0, y1 + 1):
        last = dt.date(y, 12, 28).isocalendar()[1]
        out += [f"{y}W{w}" for w in range(1, last + 1)]
    return out


# ---------------------------------------------------------------------------

def cmd_system(s, api):
    info = get(s, f"{api}/system/info")
    keep = {k: info.get(k) for k in ("version", "revision", "serverDate",
                                     "lastAnalyticsTableGeneration",
                                     "lastAnalyticsTableRuntime", "calendar")}
    save("system", f"info_{dt.date.today()}.json", keep)
    print(json.dumps(keep, indent=1))


def cmd_find_elements(s, api, term):
    fields = ("id,name,shortName,code,created,lastUpdated,valueType,domainType,"
              "aggregationType,categoryCombo[id,name],"
              "dataSetElements[dataSet[id,name,periodType]]")
    r = get(s, f"{api}/dataElements", {"filter": f"name:ilike:{term}",
                                       "fields": fields, "paging": "false"})
    save("metadata", f"elements_{term}.json", r)
    for de in r["dataElements"]:
        ds = "; ".join(f"{x['dataSet']['name']} ({x['dataSet']['periodType']})"
                       for x in de.get("dataSetElements", []))
        print(f"{de['id']}  created {de['created'][:10]}  {de['name']}  ->  {ds}")


def cmd_element(s, api, de_uid):
    de = get(s, f"{api}/dataElements/{de_uid}", {"fields": "*,categoryCombo[id,name,categoryOptionCombos[id,name]]"})
    save("metadata", f"element_{de_uid}.json", de)
    for x in de.get("dataSetElements", []):
        ds_uid = x["dataSet"]["id"]
        ds = get(s, f"{api}/dataSets/{ds_uid}",
                 {"fields": "id,name,periodType,created,lastUpdated,timelyDays,"
                            "expiryDays,openFuturePeriods,organisationUnits~size"})
        save("metadata", f"dataset_{ds_uid}.json", ds)
        print(json.dumps(ds, indent=1))


def cmd_orgunits(s, api, level):
    lv = get(s, f"{api}/organisationUnitLevels", {"fields": "level,name", "paging": "false"})
    save("metadata", "orgunit_levels.json", lv)
    r = get(s, f"{api}/organisationUnits",
            {"level": level, "paging": "false",
             "fields": "id,name,code,level,path,openingDate,closedDate,parent[id,name]"})
    ous = r["organisationUnits"]
    save("metadata", f"orgunits_level{level}.json", ous)
    with open(OUT / "metadata" / f"orgunits_level{level}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "name", "code", "province_uid", "province", "opening", "closed"])
        for o in ous:
            p = o.get("parent", {})
            w.writerow([o["id"], o["name"], o.get("code", ""), p.get("id", ""), p.get("name", ""),
                        o.get("openingDate", "")[:10], (o.get("closedDate") or "")[:10]])
    print(len(ous), "org units at level", level, "| levels:", lv)


def cmd_analytics(s, api, de_uid, ds_uid, level, y0, y1):
    """Cases and reporting metrics per zone per ISO week, one request per year."""
    dx = [de_uid] + [f"{ds_uid}.{m}" for m in METRICS]
    rows = []
    for y in range(int(y0), int(y1) + 1):
        pe = ";".join(iso_weeks(y, y))
        r = get(s, f"{api}/analytics", [
            ("dimension", "dx:" + ";".join(dx)),
            ("dimension", "pe:" + pe),
            ("dimension", f"ou:LEVEL-{level}"),
            ("outputIdScheme", "UID"),
            ("skipMeta", "false"),
        ])
        save("analytics", f"raw_{de_uid}_{y}.json", r)
        hdr = [h["name"] for h in r["headers"]]
        rows += [dict(zip(hdr, row)) for row in r["rows"]]
        print(y, len(r["rows"]), "rows")
    path = OUT / "analytics" / f"long_{de_uid}_{ds_uid}_L{level}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dx", "pe", "ou", "value"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("dx", "pe", "ou", "value")})
    print("wrote", path, len(rows), "rows")


def cmd_snapshot(s, api, de_uid, ds_uid):
    """Raw data values with created/lastUpdated, for vintage reconstruction.

    Run weekly (e.g. by cron) from now on: each run is a dated snapshot, so
    real-time forecast evaluation becomes possible even if the server keeps
    no usable audit trail."""
    today = dt.date.today().isoformat()
    roots = get(s, f"{api}/organisationUnits", {"level": 1, "fields": "id", "paging": "false"})
    root = roots["organisationUnits"][0]["id"]
    start = (dt.date.today() - dt.timedelta(weeks=60)).isoformat()
    r = get(s, f"{api}/dataValueSets", {
        "dataSet": ds_uid, "orgUnit": root, "children": "true",
        "startDate": start, "endDate": today, "includeDeleted": "true",
    })
    vals = [v for v in r.get("dataValues", []) if v["dataElement"] == de_uid]
    save(f"snapshots/{today}", f"values_{de_uid}.json", vals)
    print(len(vals), "values; window", start, "to", today)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    s, api = session()
    cmd, args = sys.argv[1], sys.argv[2:]
    {"system": cmd_system, "find-elements": cmd_find_elements, "element": cmd_element,
     "orgunits": cmd_orgunits, "analytics": cmd_analytics,
     "snapshot": cmd_snapshot}[cmd](s, api, *args)
