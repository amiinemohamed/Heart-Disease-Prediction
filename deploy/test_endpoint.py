"""Smoke test for the API, locally or on Azure.

    python deploy/test_endpoint.py                                  # http://127.0.0.1:8000
    python deploy/test_endpoint.py http://<fqdn>:8000               # Azure Container Instances
"""
import sys

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")

patients = {
    "likely low risk": {"Age": 40, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 120, "Cholesterol": 200,
                        "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 172, "ExerciseAngina": "N",
                        "Oldpeak": 0.0, "ST_Slope": "Up"},
    "likely high risk": {"Age": 63, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150, "Cholesterol": 0,
                         "FastingBS": 1, "RestingECG": "ST", "MaxHR": 105, "ExerciseAngina": "Y",
                         "Oldpeak": 2.5, "ST_Slope": "Flat"},
}

print("GET  /health ->", requests.get(f"{BASE}/health", timeout=10).json())
print("GET  /info   ->", requests.get(f"{BASE}/info", timeout=10).json())
for name, p in patients.items():
    r = requests.post(f"{BASE}/predict", json=p, timeout=10)
    print(f"POST /predict ({name}) -> {r.status_code} {r.json()}")

bad = {**patients["likely low risk"], "Sex": "X"}
r = requests.post(f"{BASE}/predict", json=bad, timeout=10)
print(f"POST /predict (invalid Sex) -> {r.status_code} (expected 422)")
