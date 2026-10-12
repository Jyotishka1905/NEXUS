from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_endpoints():
    print("Testing NEXUS API endpoints...")

    # Root
    r0 = client.get("/")
    assert r0.status_code == 200, f"Root failed: {r0.status_code}"
    print("  GET / -> 200 OK")

    # Health
    r1 = client.get("/api/health")
    assert r1.status_code == 200, f"Health failed: {r1.status_code}"
    print("  GET /api/health -> 200 OK")

    # Models benchmark
    r2 = client.get("/api/models")
    assert r2.status_code == 200, f"Models failed: {r2.status_code}"
    data2 = r2.json()
    print(f"  GET /api/models -> 200 OK ({len(data2['models'])} models listed)")

    # Regions
    r3 = client.get("/api/regions")
    assert r3.status_code == 200, f"Regions failed: {r3.status_code}"
    data3 = r3.json()
    print(f"  GET /api/regions -> 200 OK ({data3['count']} regions)")

    # Alerts
    r4 = client.get("/api/alerts")
    assert r4.status_code == 200, f"Alerts failed: {r4.status_code}"
    data4 = r4.json()
    print(f"  GET /api/alerts -> 200 OK ({data4['count']} active regional alerts)")

    # Predict
    r5 = client.post(
        "/api/forecast/predict",
        json={
            "region_name": "Brahmaputra Valley",
            "country": "India",
            "latitude": 26.20,
            "longitude": 92.93,
            "precip_7day_total_mm": 380.0,
            "precip_max_daily_mm": 110.0,
            "precip_last_1day_mm": 70.0,
            "lead_time_days": 2.0,
        },
    )
    assert r5.status_code == 200, f"Predict failed: {r5.status_code}"
    data5 = r5.json()
    print(f"  POST /api/forecast/predict -> 200 OK (Risk Level: {data5['risk_level']}, Prob: {data5['flood_probability']})")

    print("\nAll API tests PASSED successfully!")

if __name__ == "__main__":
    test_endpoints()
