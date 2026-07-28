from geopy.distance import geodesic

THRESHOLDS_KM = [1, 25, 200, 750, 2500]


def distance_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    return geodesic((a_lat, a_lon), (b_lat, b_lon)).km


def threshold_hits(dist_km: float) -> dict:
    return {f"{t}km": dist_km <= t for t in THRESHOLDS_KM}
