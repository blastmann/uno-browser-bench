# Local predictor MVP

This server is a deterministic baseline for the Edge extension. It listens only on `127.0.0.1` and makes no external model calls.

```powershell
python .\local_predictor\server.py
Invoke-RestMethod http://127.0.0.1:8765/healthz
```

The extension POSTs a redacted Browser Observation to `/v1/predict`. Replace `predict()` with a NanoJev or decider adapter only after measuring it on the held-out browser-intent data. Do not expose this endpoint beyond localhost.
