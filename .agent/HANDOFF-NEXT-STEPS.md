# Milestone 5 Next Steps (New Machine)

## 1) Clone and Environment

```bash
git clone git@github.com:Almalgo/ZK-LLM-Turbo.git
cd ZK-LLM-Turbo
python3 -m venv split-inference-env
source split-inference-env/bin/activate
pip install -r requirements.txt
```

## 2) Confirm Milestone 5 Files

```bash
ls snet_service
ls scripts/m5_snet_*.py
ls Reports/Milestone5.md
```

## 3) Materialize Runtime Config (Sepolia)

```bash
cp snet_service/snetd.config.sepolia.template.json snet_service/snetd.config.sepolia.json
```

Fill placeholders in `snet_service/snetd.config.sepolia.json`:

- `<ORG_ID>`
- `<SERVICE_ID>`
- `<DOMAIN>`
- `<SEPOLIA_RPC_URL>`

## 4) Start Backend and Daemon

Backend:

```bash
source split-inference-env/bin/activate
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

Daemon (separate shell):

```bash
snetd -c snet_service/snetd.config.sepolia.json
```

## 5) Run Milestone 5 Evidence Scripts Against Daemon

```bash
source split-inference-env/bin/activate
python scripts/m5_snet_smoke.py --base-url "http://<daemon-host>:7000" --output benchmarks/results/m5_snet_smoke_sepolia.json
python scripts/m5_snet_reliability.py --base-url "http://<daemon-host>:7000" --reliability-output benchmarks/results/m5_reliability_sepolia.json --recovery-output benchmarks/results/m5_recovery_sepolia.json
```

## 6) Update Report and Push

Update `Reports/Milestone5.md` with:

- final Sepolia evidence results
- final public service link (after mainnet publication)

Then commit and push.
