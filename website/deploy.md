# Deploy: GraphCast-lite Weather на VPS

## Требования
- Ubuntu 22.04+, 10 ГБ диск, без GPU
- Python 3.10+

## 1. Подготовка

```bash
# На VPS
sudo mkdir -p /opt/graphcast-lite
sudo chown $USER:$USER /opt/graphcast-lite

# Venv
python3 -m venv /opt/graphcast-venv
source /opt/graphcast-venv/bin/activate

# Зависимости прогноза
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install xarray cfgrib eccodes scipy scikit-learn joblib requests numpy matplotlib

# Зависимости сервера
pip install fastapi uvicorn
```

## 2. Копирование файлов

С локальной машины через SOCKS5 прокси:

```bash
PROXY="nc -X 5 -x 192.168.1.1:1080 %h %p"
VPS="root@185.42.163.63"

rsync -avz -e "ssh -o 'ProxyCommand=$PROXY'" \
  experiments/multires_nores_freeze6/config.json \
  experiments/multires_nores_freeze6/best_model.pth \
  $VPS:/opt/graphcast-lite/experiments/multires_nores_freeze6/

rsync -avz -e "ssh -o 'ProxyCommand=$PROXY'" \
  live_runtime_bundle/ \
  $VPS:/opt/graphcast-lite/live_runtime_bundle/

rsync -avz -e "ssh -o 'ProxyCommand=$PROXY'" \
  scripts/live_gdas_forecast.py \
  $VPS:/opt/graphcast-lite/scripts/

rsync -avz -e "ssh -o 'ProxyCommand=$PROXY'" \
  src/ $VPS:/opt/graphcast-lite/src/

rsync -avz -e "ssh -o 'ProxyCommand=$PROXY'" \
  website/ $VPS:/opt/graphcast-lite/website/

scp -o "ProxyCommand=$PROXY" requirements.txt $VPS:/opt/graphcast-lite/
```

## 3. Systemd сервис (uvicorn)

```bash
sudo tee /etc/systemd/system/graphcast-web.service << 'EOF'
[Unit]
Description=GraphCast-lite Weather Web
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/graphcast-lite
ExecStart=/opt/graphcast-venv/bin/uvicorn website.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable graphcast-web
sudo systemctl start graphcast-web
```

## 4. Cron (каждые 6 часов)

```bash
# Дать права
chmod +x /opt/graphcast-lite/website/cron_forecast.sh

# Crontab
crontab -e
# Добавить:
0 1,7,13,19 * * * /opt/graphcast-lite/website/cron_forecast.sh >> /var/log/graphcast-forecast.log 2>&1
```

## 5. Первый запуск (вручную)

```bash
cd /opt/graphcast-lite
/opt/graphcast-lite/website/cron_forecast.sh 2>&1 | tee /var/log/graphcast-forecast.log
```

## 6. Проверка

```bash
# Сервер работает?
curl http://localhost:8000/api/status

# Прогноз есть?
curl http://localhost:8000/api/forecast | python3 -m json.tool | head -20

# Лог
tail -50 /var/log/graphcast-forecast.log

# Диск
df -h /
du -sh /opt/graphcast-lite/results/*
```

## 7. Nginx (опционально, reverse proxy)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Размеры (ориентир)

| Компонент | Размер |
|---|---|
| PyTorch CPU + deps | ~1.5 ГБ |
| Модель + bundle + код | ~100 МБ |
| forecast.pt × 2 | ~740 МБ |
| forecast.json | ~200-500 КБ |
| **Итого** | **~2.4 ГБ** |

Свободно должно оставаться ≥3 ГБ для временных GDAS grib2 файлов (~700 МБ).
