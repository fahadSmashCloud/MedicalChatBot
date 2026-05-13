# Deployment Guide

---

## Option 1 — Streamlit Community Cloud (Easiest, Free)

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Select repo, branch `main`, main file `medibot.py`.
4. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ADZUNA_APP_ID = "..."
   ADZUNA_APP_KEY = "..."
   RAPIDAPI_KEY = "..."
   ```
5. Click **Deploy**. Done — public HTTPS URL in ~2 minutes.

Limitation: 1 GB RAM on free tier. The FAISS index + sentence-transformer model (~90 MB) fit comfortably.

---

## Option 2 — Docker (Self-hosted or any VPS)

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install build tools needed for faiss-cpu and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Disable file watcher — avoids watchdog issues in containers
CMD ["streamlit", "run", "medibot.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--server.headless=true"]
```

### Build and run

```bash
# Build
docker build -t ai-workbench .

# Run with env vars
docker run -p 8501:8501 \
  -e GROQ_API_KEY=gsk_... \
  -e ADZUNA_APP_ID=... \
  -e ADZUNA_APP_KEY=... \
  -e RAPIDAPI_KEY=... \
  ai-workbench

# Or use a .env file
docker run -p 8501:8501 --env-file .env ai-workbench
```

Open `http://localhost:8501`.

### docker-compose.yml

```yaml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      # Persist the FAISS index and roadmap across restarts
      - ./vectorstore:/app/vectorstore
      - ./data:/app/data
    restart: unless-stopped
```

```bash
docker compose up -d
docker compose logs -f
```

---

## Option 3 — AWS EC2 / GCP Compute Engine / Azure VM

```bash
# On a fresh Ubuntu 22.04 instance (t3.medium or equivalent — 2 vCPU, 4 GB RAM)

# Install Python 3.11
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git

# Clone repo
git clone <repo-url>
cd MedicalChatBot

# Create venv
python3.11 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env
echo "GROQ_API_KEY=gsk_..." > .env

# Run with nohup so it survives SSH disconnect
nohup streamlit run medibot.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.fileWatcherType none \
  --server.headless true \
  > app.log 2>&1 &

echo "Running at http://$(curl -s ifconfig.me):8501"
```

Open port 8501 in your security group / firewall rules.

### Keep alive with systemd

```ini
# /etc/systemd/system/ai-workbench.service
[Unit]
Description=AI Workbench Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/MedicalChatBot
EnvironmentFile=/home/ubuntu/MedicalChatBot/.env
ExecStart=/home/ubuntu/MedicalChatBot/.venv/bin/streamlit run medibot.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.fileWatcherType none \
  --server.headless true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-workbench
sudo systemctl start ai-workbench
sudo systemctl status ai-workbench
```

---

## Option 4 — Railway / Render / Fly.io

These platforms auto-detect Python and deploy from `requirements.txt`.

**Start command** (set in dashboard):
```
streamlit run medibot.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none
```

Add environment variables in the platform's secrets/env section.

Recommended plan: at least 512 MB RAM (1 GB preferred for sentence-transformer model loading).

---

## HTTPS / Reverse Proxy with nginx

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
```

Then use Certbot for free TLS:
```bash
sudo certbot --nginx -d yourdomain.com
```

---

## FAISS Index Persistence

The FAISS index lives at `vectorstore/db_faiss/`. In containerised deployments, mount it as a volume so ingested PDFs survive restarts:

```bash
docker run -v $(pwd)/vectorstore:/app/vectorstore ...
```

---

## Resource Requirements

| Component | RAM | Notes |
|-----------|-----|-------|
| Streamlit app | ~150 MB | Base |
| sentence-transformers model | ~90 MB | Loaded once, cached |
| FAISS index | Varies | ~1 MB per 5,000 chunks |
| **Minimum recommended** | **512 MB** | |
| **Comfortable** | **1–2 GB** | Handles concurrent users |

GPU is not required — all inference runs on CPU via Groq API (cloud) and sentence-transformers (local CPU).

---

## Production Checklist

- [ ] `GROQ_API_KEY` is set and valid
- [ ] `.env` is in `.gitignore`
- [ ] `vectorstore/` is mounted as a persistent volume (Docker) or committed (small indexes)
- [ ] Port 8501 is open or proxied via nginx
- [ ] TLS certificate is configured (Certbot recommended)
- [ ] `--server.fileWatcherType none` is in the start command (avoids watchdog issues)
- [ ] `--server.headless true` is set for server deployments
- [ ] Health check: `curl -f http://localhost:8501/_stcore/health`
