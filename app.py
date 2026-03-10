import os
import re
import json
import uuid
import base64
import shutil
import urllib.request
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template

from PIL import Image
import folium
from branca.element import MacroElement, Template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# Абсолютные пути — работает из любой директории
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'output')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ─────────────────────────────────────────────
#  Ключ Anthropic API (вставьте свой)
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

# ─────────────────────────────────────────────
#  Вырезаем левый верхний угол
# ─────────────────────────────────────────────

def crop_top_left(img: Image.Image) -> Image.Image:
    """Вырезает левую полосу (1/6 ширины × вся высота) — там вертикальный текст с координатами."""
    w, h = img.size
    piece = img.crop((0, 0, w // 6, h))
    return piece


def image_to_base64(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)  # 70 достаточно для OCR
    data = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return data


# ─────────────────────────────────────────────
#  Claude Vision OCR
# ─────────────────────────────────────────────

def _call_claude(img: Image.Image, prompt: str) -> str:
    """Базовый вызов Claude Vision API, возвращает текст ответа."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError('ANTHROPIC_API_KEY не задан')

    b64 = image_to_base64(img)
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 256,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    }).encode()

    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=payload,
        headers={'Content-Type': 'application/json', 'x-api-key': ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    if 'error' in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data['content'][0]['text'].strip()


def _parse_coords(text: str):
    """Парсит два числа из ответа Claude. Возвращает (lat, lon) или None."""
    m = re.search(r'(-?\d{1,3}[.,]\d+)[^\d\-]+(-?\d{1,3}[.,]\d+)', text)
    if m:
        lat = float(m.group(1).replace(',', '.'))
        lon = float(m.group(2).replace(',', '.'))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    return None


def ocr_with_claude(img: Image.Image) -> tuple:
    """Режим 1: левая полоса с Широта/Долгота (вертикальный текст)."""
    prompt = (
        "На изображении есть текст с GPS координатами. Найди строки с координатами.\n"
        "Возможные форматы:\n"
        "- 'Широта: 42.XXXXXX' и 'Долгота: 133.XXXXXX' (русские метки)\n"
        "- просто числа вида 42.XXXXXX и 133.XXXXXX\n"
        "КРИТИЧНО: читай каждую цифру отдельно — не угадывай, не округляй.\n"
        "Цифры 0, 1, 6, 7, 9 — легко путаются, смотри внимательно на каждую.\n"
        "Широта: ~42-43 (два знака до точки). Долгота: ~130-135 (три знака до точки).\n"
        "Ответь ТОЛЬКО двумя числами через запятую, например: 42.731609, 133.016745\n"
        "Никакого другого текста."
    )
    text = _call_claude(img, prompt)
    if 'NOT_FOUND' in text:
        return None
    return _parse_coords(text)


def ocr_bottom_text(img: Image.Image) -> tuple:
    """Режим 2: нижняя строка формата 42.82492°C 132.88594°E (внизу фото)."""
    prompt = (
        "В нижней части изображения есть мелкий белый текст с GPS координатами.\n"
        "Формат: ХХ.ХХХХ°C ХХХ.ХХХХ°E  (где C означает North, E означает East)\n"
        "Прочитай оба числа максимально точно — каждая цифра важна.\n"
        "Ответь ТОЛЬКО двумя числами через запятую, например: 42.82492, 132.88594\n"
        "Только числа, никакого другого текста."
    )
    text = _call_claude(img, prompt)
    if 'NOT_FOUND' in text:
        return None
    return _parse_coords(text)


def ocr_universal(img: Image.Image) -> tuple:
    """Универсальный режим: отправляет полное фото, Claude сам находит координаты."""
    prompt = (
        "На этом фото есть GPS координаты. Они могут быть:\n"
        "1. Вертикальный текст вдоль левого края: 'Широта: ХХ.ХХХХ' и 'Долгота: ХХХ.ХХХХ'\n"
        "2. Мелкий белый текст внизу: 'ХХ.ХХХХ°C ХХХ.ХХХХ°E'\n"
        "КРИТИЧНО: читай каждую цифру отдельно — не угадывай, не округляй.\n"
        "Цифры 0, 1, 6, 7, 9 — легко путаются, смотри внимательно на каждую.\n"
        "Широта: ~42-43 (два знака до точки). Долгота: ~130-135 (три знака до точки).\n"
        "Найди координаты и ответь ТОЛЬКО двумя числами через запятую.\n"
        "Пример: 42.82516, 132.88607\n"
        "Только числа — никакого другого текста."
    )
    text = _call_claude(img, prompt)
    if 'NOT_FOUND' in text:
        return None
    return _parse_coords(text)


def ocr_image(path: str, name: str) -> dict:
    """Автодетекция формата: один вызов Claude на полное фото."""
    try:
        with Image.open(path) as img_orig:
            img_orig.load()
            w, h = img_orig.size

            # Шаг 1: Левая полоса — кроп w//5, ROTATE_90, upscale x2 (не x4!)
            strip = img_orig.crop((0, 0, w // 5, h)).transpose(Image.ROTATE_90)
            strip = strip.resize((strip.width * 2, strip.height * 2), Image.LANCZOS)
            coords = ocr_with_claude(strip)
            del strip
            if coords:
                return {'name': name, 'raw': 'Claude Vision API (left)', 'coords': list(coords), 'ok': True}

            # Шаг 2: Нижняя строка (°C °E формат)
            bottom = img_orig.crop((0, int(h * 0.87), w, h))
            bottom = bottom.resize((bottom.width * 2, bottom.height * 2), Image.LANCZOS)
            coords = ocr_bottom_text(bottom)
            del bottom
            if coords:
                return {'name': name, 'raw': 'Claude Vision API (bottom)', 'coords': list(coords), 'ok': True}

            # Шаг 3: Полное фото — ресайз до 1200px по длинной стороне
            max_side = 1200
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img_small = img_orig.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            else:
                img_small = img_orig.copy()

        coords = ocr_universal(img_small)
        del img_small
        if coords:
            return {'name': name, 'raw': 'Claude Vision API (full)', 'coords': list(coords), 'ok': True}

        return {'name': name, 'raw': 'Координаты не найдены', 'coords': None, 'ok': False}
    except Exception as e:
        return {'name': name, 'raw': str(e), 'coords': None, 'ok': False}



# ─────────────────────────────────────────────
#  Генерация карты (логика из make_map.py)
# ─────────────────────────────────────────────

ROUTE_JS = Template("""
{% macro script(this, kwargs) %}
(function() {
  var map = {{ this._parent.get_name() }};
  var selected = [];
  var activeLine = null;
  var activePopup = null;

  function fmtMeters(m) {
    if (m >= 1000) return (m/1000).toFixed(3) + " km";
    return m.toFixed(1) + " m";
  }

  var offsetLat = 0.00012;
  var offsetLng = 0.00010;

  var SelectedIcon = L.DivIcon.extend({options: {className: ''}});
  var selectedIcon = new SelectedIcon({
    html: "<div style='width:14px;height:14px;border-radius:50%;background:#ff3b30;border:2px solid white;box-shadow:0 0 6px rgba(0,0,0,.35)'></div>",
    iconSize: [14, 14],
    iconAnchor: [7, 7]
  });

  function getMarkerName(marker) {
    if (marker.getTooltip && marker.getTooltip()) return marker.getTooltip().getContent();
    if (marker.getPopup && marker.getPopup()) return marker.getPopup().getContent();
    return "Point";
  }

  function reset() {
    selected.forEach(function(s) {
      if (s.marker && s.marker.__origIcon) s.marker.setIcon(s.marker.__origIcon);
    });
    selected = [];
    if (activeLine) { map.removeLayer(activeLine); activeLine = null; }
    if (activePopup) { map.closePopup(activePopup); activePopup = null; }
  }

  function updateLineAndPopup(lastLatLng) {
    if (activeLine) { map.removeLayer(activeLine); activeLine = null; }
    if (activePopup) { map.closePopup(activePopup); activePopup = null; }
    if (selected.length < 2) return;

    var latlngs = selected.map(function(s) { return s.latlng; });
    activeLine = L.polyline(latlngs, {weight: 4, opacity: 0.9}).addTo(map);

    var total = 0;
    for (var i=1; i<latlngs.length; i++) total += latlngs[i-1].distanceTo(latlngs[i]);
    var last = latlngs[latlngs.length-2].distanceTo(latlngs[latlngs.length-1]);

    var aName = selected[selected.length-2].name;
    var bName = selected[selected.length-1].name;

    var body = "<b>" + aName + " → " + bName + "</b><br>"
             + "<b>Last:</b> " + fmtMeters(last) + "<br>"
             + "<b>Total:</b> " + fmtMeters(total);

    var base = lastLatLng || latlngs[latlngs.length-1];
    var shifted = L.latLng(base.lat + offsetLat, base.lng + offsetLng);

    activePopup = L.popup({closeButton: true, autoPan: true, autoPanPadding: [20,20]})
      .setLatLng(shifted).setContent(body).openOn(map);
  }

  var ResetControl = L.Control.extend({
    options: { position: 'topright' },
    onAdd: function() {
      var div = L.DomUtil.create('div', 'leaflet-bar');
      div.style.cssText = 'background:rgba(255,255,255,.92);padding:6px 10px;border-radius:10px;border:1px solid #999;font-size:12px;cursor:pointer;user-select:none';
      div.innerHTML = "<b>Route distance</b><br>Tap markers to build route<br><span style='opacity:.7'>Reset</span>";
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.on(div, 'click', function() { reset(); });
      return div;
    }
  });
  map.addControl(new ResetControl());
  map.on('contextmenu', function() { reset(); });

  function attachToMarkers() {
    map.eachLayer(function(layer) {
      if (layer instanceof L.Marker) {
        if (layer.__route_hooked) return;
        layer.__route_hooked = true;
        if (!layer.__origIcon) layer.__origIcon = layer.options.icon;
        layer.on('click', function(e) {
          var name = getMarkerName(layer);
          layer.setIcon(selectedIcon);
          selected.push({latlng: e.latlng, name: name, marker: layer});
          updateLineAndPopup(e.latlng);
        });
      }
    });
  }

  attachToMarkers();
  map.whenReady(function(){ attachToMarkers(); });
})();
{% endmacro %}
""")


def build_map(points: list, out_path: str):
    """points = [(lat, lon, name), ...]"""
    avg_lat = sum(p[0] for p in points) / len(points)
    avg_lon = sum(p[1] for p in points) / len(points)

    m = folium.Map(location=(avg_lat, avg_lon), zoom_start=17, control_scale=True)

    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # metres
        φ1, φ2 = radians(lat1), radians(lat2)
        dφ = radians(lat2 - lat1)
        dλ = radians(lon2 - lon1)
        a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    for i, (lat, lon, name) in enumerate(points):
        # Check if any other point is within 15m
        too_close = any(
            haversine(lat, lon, lat2, lon2) < 15
            for j, (lat2, lon2, _) in enumerate(points) if j != i
        )
        marker_color = 'red' if too_close else 'blue'
        folium.Marker(
            (lat, lon),
            tooltip=name,
            popup=f"{name}<br>{lat}, {lon}",
            icon=folium.Icon(color=marker_color, icon='circle', prefix='fa'),
        ).add_to(m)

    m.fit_bounds([(p[0], p[1]) for p in points])

    class RouteDistance(MacroElement):
        _template = ROUTE_JS

    m.add_child(RouteDistance())
    m.save(out_path)


# ─────────────────────────────────────────────
#  API routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process():
    from flask import Response, stream_with_context
    files = request.files.getlist('photos')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    # Сохраняем все файлы сразу (до стриминга)
    session_id = uuid.uuid4().hex
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename:
            continue
        save_path = os.path.join(session_dir, f.filename)
        f.save(save_path)
        saved.append((Path(f.filename).stem, save_path))

    def generate():
        results = []
        total = len(saved)
        for i, (name, path) in enumerate(saved):
            yield f"data: {json.dumps({'type':'status','step':'ocr','i':i+1,'total':total,'name':name})}\n\n"
            try:
                result = ocr_image(path, name)
            except Exception as e:
                result = {'name': name, 'raw': str(e), 'coords': None, 'ok': False}
            results.append(result)
            yield f"data: {json.dumps({'type':'result','entry':result})}\n\n"

        shutil.rmtree(session_dir, ignore_errors=True)
        yield f"data: {json.dumps({'type':'done','results':results,'session_id':session_id})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ─────────────────────────────────────────────
#  Downloader: ссылки → прямые URL → скачать
# ─────────────────────────────────────────────

import urllib.error
import threading

UA = ("Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

OG_RE  = re.compile(r'<meta[^>]+property=["\'"]og:image["\'"][^>]+content=["\'"]([^"\'"]+)["\'"]>', re.I)
IBB_RE = re.compile(r"https?://i\.ibb\.co/[^\s\"']+", re.I)


def resolve_direct_url(share_url: str) -> str:
    req = urllib.request.Request(share_url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        html = r.read().decode("utf-8", errors="ignore")
    m = OG_RE.search(html)
    if m:
        return m.group(1)
    m = IBB_RE.search(html)
    if m:
        return m.group(0)
    if re.search(r"\.(jpg|jpeg|png|webp)(\?.*)?$", share_url, re.I):
        return share_url
    raise RuntimeError("Не удалось найти прямую ссылку: " + share_url)


def download_image(url: str, dest_path: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        with open(dest_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def parse_links(text: str) -> list:
    # Extract all URLs starting with https anywhere in text
    raw = re.findall(r'https?://\S+', text)
    # Clean trailing punctuation
    links = [re.sub(r'[.,;!?)\]>"\']+$', '', u) for u in raw]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for l in links:
        if l not in seen:
            seen.add(l)
            result.append(l)
    return result


# Хранилище прогресса по session_id
_sessions: dict = {}
_sessions_lock = __import__('threading').Lock()


@app.route("/api/process_links", methods=["POST"])
def process_links():
    from flask import Response, stream_with_context
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No links provided"}), 400
    links = parse_links(text)
    if not links:
        return jsonify({"error": "No valid URLs found"}), 400

    session_id = uuid.uuid4().hex
    session_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    os.makedirs(session_dir, exist_ok=True)

    def generate():
        import queue as _queue
        import threading as _threading
        import random as _random

        total = len(links)
        results_ordered = [None] * total
        progress_q = _queue.Queue()
        sem = _threading.Semaphore(2)

        WORKER_TIMEOUT = 30  # секунд на одно фото
        MAX_RETRIES = 2

        def process_one(i, link, attempt=1):
            slug = link.rstrip("/").split("/")[-1]
            name = slug + ".jpg"
            entry = {"name": name, "link": link, "idx": i+1, "ok": False, "raw": "", "coords": None}
            progress_q.put({"type": "status", "step": "download", "i": i+1, "total": total,
                            "name": name, "attempt": attempt})
            try:
                direct = resolve_direct_url(link)
                m = re.search(r"\.(jpg|jpeg|png|webp)", direct, re.I)
                ext = ("." + m.group(1).lower()) if m else ".jpg"
                dest = os.path.join(session_dir, name + ext)
                download_image(direct, dest)
                progress_q.put({"type": "status", "step": "ocr", "i": i+1, "total": total,
                                "name": name, "attempt": attempt})
                result = ocr_image(dest, name)
                entry.update(result)
                try:
                    os.remove(dest)
                except Exception:
                    pass
            except Exception as e:
                entry["raw"] = str(e)
            return entry

        def run_worker(i, link, attempt=1):
            # Семафор берём только на время активной работы, не на таймаут
            with sem:
                # Запускаем реальную работу в отдельном потоке с таймаутом
                result_box = [None]
                def _run():
                    result_box[0] = process_one(i, link, attempt)
                inner = _threading.Thread(target=_run, daemon=True)
                inner.start()
                inner.join(timeout=WORKER_TIMEOUT)

            # Семафор уже отпущен — теперь ждём результат
            if inner.is_alive():
                slug = link.rstrip("/").split("/")[-1]
                name = slug + ".jpg"
                entry = {"name": name, "link": link, "idx": i+1, "ok": False,
                         "raw": "timeout", "coords": None}
            else:
                entry = result_box[0] or {"name": link.rstrip("/").split("/")[-1]+".jpg",
                                           "link": link, "idx": i+1, "ok": False,
                                           "raw": "no result", "coords": None}
            results_ordered[i] = entry
            progress_q.put({"type": "result", "entry": entry})

        # ── Основной проход ──────────────────────────────────────────
        indexed_links = list(enumerate(links))
        _random.shuffle(indexed_links)

        threads = []
        for i, link in indexed_links:
            t = _threading.Thread(target=run_worker, args=(i, link, 1), daemon=True)
            threads.append(t)
            t.start()

        done_count = 0
        while done_count < total:
            msg = progress_q.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] == "result":
                done_count += 1

        # ── Retry для failed/timeout ─────────────────────────────────
        for attempt in range(2, MAX_RETRIES + 2):
            retry_list = [(i, links[i]) for i in range(total)
                          if results_ordered[i] and not results_ordered[i].get("ok")]
            if not retry_list:
                break

            yield f"data: {json.dumps({'type': 'retry_start', 'attempt': attempt, 'count': len(retry_list)})}\n\n"

            retry_threads = []
            for i, link in retry_list:
                t = _threading.Thread(target=run_worker, args=(i, link, attempt), daemon=True)
                retry_threads.append(t)
                t.start()

            for _ in range(len(retry_list)):
                msg = progress_q.get()
                yield f"data: {json.dumps(msg)}\n\n"

        results = [r for r in results_ordered if r is not None]
        for r in results:
            if not r.get("ok") and not r.get("raw"):
                r["raw"] = "failed"
        shutil.rmtree(session_dir, ignore_errors=True)
        yield f"data: {json.dumps({'type':'done','results':results,'session_id':session_id})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/build_map', methods=['POST'])
def build_map_route():
    data = request.get_json()
    points_raw = data.get('points', [])

    if not points_raw:
        return jsonify({'error': 'No points provided'}), 400

    points = []
    for p in points_raw:
        try:
            points.append((float(p['lat']), float(p['lon']), str(p['name'])))
        except Exception:
            continue

    if not points:
        return jsonify({'error': 'Invalid points format'}), 400

    map_id = uuid.uuid4().hex
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'map_{map_id}.html')
    build_map(points, out_path)

    return jsonify({'map_id': map_id})


@app.route('/map/<map_id>')
def view_map(map_id):
    if not re.fullmatch(r'[0-9a-f]{32}', map_id):
        return 'Not found', 404
    path = os.path.join(app.config['OUTPUT_FOLDER'], f'map_{map_id}.html')
    if not os.path.exists(path):
        return 'Map not found', 404
    return send_file(path)


@app.route('/api/share', methods=['POST'])
def share_map():
    data = request.get_json()
    map_id = data.get('map_id', '')
    if not re.fullmatch(r'[0-9a-f]{32}', map_id):
        return jsonify({'error': 'Invalid map_id'}), 400

    path = os.path.join(app.config['OUTPUT_FOLDER'], f'map_{map_id}.html')
    if not os.path.exists(path):
        return jsonify({'error': 'Map not found'}), 404

    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    payload = json.dumps({
        'html': html_content,
        'ttl': '3d',
        'fileName': f'gps_map_{map_id[:8]}.html',
        'visibility': 'private'
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://pagedrop.io/api/upload',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        if not result.get('success'):
            code = result.get('code', '')
            err  = result.get('error', str(result))
            return jsonify({'error': f'{code}: {err}'}), 502

        url = result['data']['url']
        return jsonify({'url': url, 'service': 'pagedrop.io'})

    except Exception as e:
        return jsonify({'error': str(e)}), 502


if __name__ == '__main__':
    app.run(debug=True, port=5000)
