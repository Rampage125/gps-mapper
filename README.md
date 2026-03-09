# GPS Mapper 🗺️

Веб-приложение: загружаете фото → Claude Vision API читает координаты → интерактивная карта.

## Быстрый старт

### 1. Получить Anthropic API ключ
1. https://console.anthropic.com → API Keys → Create Key
2. Скопировать ключ (начинается с `sk-ant-...`)

### 2. Установить зависимости

**Termux:**
```bash
pkg install python -y
pip install flask pillow folium branca
```

**macOS / Linux / Windows:**
```bash
pip install -r requirements.txt
```

### 3. Запустить

**Termux / Linux / macOS:**
```bash
export ANTHROPIC_API_KEY="sk-ant-ваш-ключ"
python app.py
```

**Windows:**
```cmd
set ANTHROPIC_API_KEY=sk-ant-ваш-ключ
python app.py
```

**Или вставить ключ прямо в app.py строка 26:**
```python
ANTHROPIC_API_KEY = 'sk-ant-ваш-ключ'
```

### 4. Открыть браузер
```
http://localhost:5000
```
