"""Storm metrics in reliability terms: customer-hours, the storm's contribution to SAIDI, CAIDI, the long tail of
customers out beyond 24, 72 and 96 hours, the worst county, and the crews and crew-hours of the restoration.

Every quantity is customer weighted, over crew jobs and automatic restorations, from the outage d_j of every job j
with c_j customers.
"""
import numpy as np

AUTO_VISIBLE = 0.625    # share of an hour's automatic restorations still out at the end of that hour (arrival uniform
                        # in the hour, duration uniform 15 to 60 minutes)

ADDITIVE = ("customer_hours", "customers_interrupted", "over_24h", "over_72h", "over_96h", "local_crew_hours",
            "aid_crew_hours", "aid_crews", "saidi_min", "busy_crew_hours", "travel_crew_hours")
AVERAGED = ("worst_county_h", "restoration_end_h", "crews_max", "caidi_h")


def storm_metrics(outcome, auto, counties, customers=None):
    """outcome: what the engine returns; auto: customers, hours and county of the automatic restorations;
    customers: customers served per county when they differ from the county table (what-if growth)."""
    d = np.concatenate([outcome["finish"] - outcome["t_arr"], auto["hours"]])
    w = np.concatenate([outcome["customers"], auto["customers"]])
    k = np.concatenate([outcome["county"], auto["county"]]).astype(int)
    K = len(counties); W = w.sum()
    cust = counties["customers"].to_numpy(float) if customers is None else np.asarray(customers, float)
    sw = np.bincount(k, w, K); sd = np.bincount(k, w * d, K)
    m = {"customer_hours": float((w * d).sum()), "customers_interrupted": float(W),
         "over_24h": float(w[d > 24].sum()), "over_72h": float(w[d > 72].sum()), "over_96h": float(w[d > 96].sum()),
         "restoration_end_h": float(outcome["t_end"])}
    mean_k = np.divide(sd, sw, out=np.full(K, np.nan), where=sw > 0)
    hit = sw >= 0.005 * cust
    m["worst_county_h"] = float(np.nanmax(np.where(hit, mean_k, np.nan))) if hit.any() else float("nan")
    # paid crew-hours: local crews from the start of the storm, aid crews from the request, to the end of restoration
    n_local = outcome.get("crews_local", 0)
    m["local_crew_hours"] = float(n_local * outcome["t_end"])
    m["aid_crew_hours"] = float(outcome.get("aid_hours", 0.0))
    m["aid_crews"] = float(outcome["aid_crews"] if outcome.get("aid_requested_at") is not None else 0)
    # reliability terms: the storm's contribution to SAIDI (IEEE Std 1366, major event days included), the mean
    # interruption duration per customer interrupted, crews on duty once all aid has joined, and the hours crews
    # spent on jobs (travel, setup and repair) and driving between counties
    m["saidi_min"] = float(60.0 * m["customer_hours"] / cust.sum())
    m["caidi_h"] = float(m["customer_hours"] / W) if W > 0 else 0.0
    m["crews_max"] = float(n_local + m["aid_crews"])
    m["busy_crew_hours"] = float(outcome["crew_hours"].sum()) if "crew_hours" in outcome else float("nan")
    m["travel_crew_hours"] = float(outcome["travel_hours"].sum()) if "travel_hours" in outcome else float("nan")
    m["county_mean_outage_h"] = mean_k
    return m


def restoration_curve(outcome, auto=None, hours=None):
    """Customers out at the end of every hour, like the observed EAGLE-I series: crew jobs plus automatic restorations."""
    t_arr, fin, c = outcome["t_arr"], outcome["finish"], outcome["customers"]
    H = int(hours or (np.ceil(np.nanmax(fin, initial=0.0)) + 2))
    a = np.zeros(H); f = np.zeros(H)
    np.add.at(a, np.clip(np.ceil(t_arr).astype(int) - 1, 0, H - 1), c)
    np.add.at(f, np.clip(np.ceil(fin).astype(int) - 1, 0, H - 1), c)
    curve = np.cumsum(a) - np.cumsum(f)
    if auto is not None and len(auto["customers"]):
        per_h = np.bincount(auto["hour"], auto["customers"], minlength=H)[:H]
        curve += AUTO_VISIBLE * per_h
    return np.clip(curve, 0, None)
