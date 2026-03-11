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

    # Anthropic API limit: 8000px max side, ~5MB. Cap at 1500px to be safe.
    MAX_DIM = 1500
    w, h = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore')
        print(f"[CLAUDE API ERROR] HTTP {e.code}: {body}", flush=True)
        raise RuntimeError(f"API {e.code}: {body}")
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
  var proximityThreshold = 15;

  function fmtMeters(m) {
    if (m >= 1000) return (m/1000).toFixed(3) + " km";
    return m.toFixed(1) + " m";
  }

  var offsetLat = 0.00012;
  var offsetLng = 0.00010;

  var SelectedIcon = L.DivIcon.extend({options: {className: ''}});
  var selectedIcon = new SelectedIcon({
    html: "<svg class='gps-pin' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 36' width='24' height='36'><path d='M12 0C5.4 0 0 5.4 0 12c0 8.4 12 24 12 24S24 20.4 24 12C24 5.4 18.6 0 12 0z' fill='#ff3b30' stroke='white' stroke-width='1'/><circle cx='12' cy='12' r='5' fill='white' opacity='0.85'/></svg>",
    iconSize: [24, 36],
    iconAnchor: [12, 36]
  });

  function getMarkerName(marker) {
    if (marker.getTooltip && marker.getTooltip()) return marker.getTooltip().getContent();
    if (marker.getPopup && marker.getPopup()) return marker.getPopup().getContent();
    return "Point";
  }

  function applyProximityColors(threshold) {
    proximityThreshold = threshold;
    var markers = [];
    map.eachLayer(function(layer) {
      if (layer instanceof L.Marker) markers.push(layer);
    });
    markers.forEach(function(m) {
      var ll = m.getLatLng();
      var tooClose = markers.some(function(other) {
        return other !== m && ll.distanceTo(other.getLatLng()) < threshold;
      });
      m.__proxColor = tooClose ? '#ff3b30' : '#4286f4';
      var el = m.getElement();
      if (el) {
        var pin = el.querySelector('.gps-pin path');
        if (pin) pin.setAttribute('fill', m.__proxColor);
      }
    });
  }

  function reset() {
    selected.forEach(function(s) {
      if (s.marker && s.marker.__origIcon) {
        s.marker.setIcon(s.marker.__origIcon);
        (function(marker) {
          setTimeout(function() {
            var el = marker.getElement();
            if (el) {
              var pin = el.querySelector('.gps-pin path');
              if (pin) pin.setAttribute('fill', marker.__proxColor || '#4286f4');
            }
          }, 0);
        })(s.marker);
      }
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

  // Min distance control (top-left)
  var ProximityControl = L.Control.extend({
    options: { position: 'topleft' },
    onAdd: function() {
      var div = L.DomUtil.create('div', 'leaflet-bar');
      div.style.cssText = 'background:rgba(255,255,255,.92);padding:6px 10px;border-radius:10px;border:1px solid #999;font-size:12px;user-select:none';
      div.innerHTML = "<b>Min distance</b><br>"
        + "<span style='opacity:.7'>closer = </span><span style='color:#ff3b30;font-weight:bold'>&#9632;</span> red<br>"
        + "<input type='number' id='prox-input' value='15' min='1' max='9999' "
        + "style='width:52px;font-size:12px;border:1px solid #ccc;border-radius:4px;"
        + "padding:2px 4px;text-align:right;margin-top:4px'> m";
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.disableScrollPropagation(div);
      setTimeout(function() {
        var inp = div.querySelector('#prox-input');
        if (inp) L.DomEvent.on(inp, 'input change', function() {
          var val = parseFloat(this.value);
          if (!isNaN(val) && val > 0) applyProximityColors(val);
        });
      }, 0);
      return div;
    }
  });
  map.addControl(new ProximityControl());

  // Route distance control (top-right)
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
    applyProximityColors(proximityThreshold);
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

    for lat, lon, name in points:
        folium.Marker(
            (lat, lon),
            tooltip=name,
            popup=f"{name}<br>{lat}, {lon}",
            icon=folium.DivIcon(
                html="<svg class='gps-pin' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 36' width='24' height='36'>"
                     "<path d='M12 0C5.4 0 0 5.4 0 12c0 8.4 12 24 12 24S24 20.4 24 12C24 5.4 18.6 0 12 0z' "
                     "fill='#4286f4' stroke='white' stroke-width='1'/>"
                     "<circle cx='12' cy='12' r='5' fill='white' opacity='0.85'/></svg>",
                icon_size=(24, 36),
                icon_anchor=(12, 36),
            ),
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

        PER_LINK_TIMEOUT = 120
        MAX_CONCURRENT = 5
        MAX_RETRIES = 2

        def worker(i, link, attempt):
            """Воркер: скачивает + OCR одну ссылку с таймаутом через Event."""
            slug = link.rstrip("/").split("/")[-1]
            name = slug + ".jpg"
            done_evt = _threading.Event()
            result_box = [None]

            def _run():
                import socket as _s2
                _s2.setdefaulttimeout(15)
                slug2 = link.rstrip("/").split("/")[-1]
                name2 = slug2 + ".jpg"
                entry = {"name": name2, "link": link, "idx": i+1, "ok": False, "raw": "", "coords": None}
                # Статус: download
                progress_q.put({"type": "status", "step": "download", "i": i+1,
                                 "total": total, "name": name2, "attempt": attempt})
                try:
                    entry["_step"] = "resolve"
                    direct = resolve_direct_url(link)
                    m = re.search(r"\.(jpg|jpeg|png|webp)", direct, re.I)
                    ext = ("." + m.group(1).lower()) if m else ".jpg"
                    dest = os.path.join(session_dir, name2 + ext)
                    entry["_step"] = "download"
                    download_image(direct, dest)
                    # Статус: ocr
                    progress_q.put({"type": "status", "step": "ocr", "i": i+1,
                                     "total": total, "name": name2, "attempt": attempt})
                    entry["_step"] = "ocr"
                    result = ocr_image(dest, name2)
                    entry.update(result)
                    try:
                        os.remove(dest)
                    except Exception:
                        pass
                except Exception as e:
                    if hasattr(e, 'read'):
                        try:
                            body = e.read().decode('utf-8', errors='ignore')
                            entry["raw"] = f"HTTP {e.code} ({entry.get('_step','?')}): {body[:250]}"
                        except Exception:
                            entry["raw"] = str(e)
                    else:
                        entry["raw"] = str(e)
                result_box[0] = entry
                done_evt.set()

            t = _threading.Thread(target=_run, daemon=True)
            t.start()
            if done_evt.wait(timeout=PER_LINK_TIMEOUT):
                return result_box[0]
            return {"name": name, "link": link, "idx": i+1, "ok": False, "raw": "timeout", "coords": None}

        def run_batch(items):
            """Запускает items через очередь с MAX_CONCURRENT воркерами.
            Статусы и результаты читаются из общей progress_q.
            Keepalive-комментарии шлются каждые 3 сек чтобы прокси не резал соединение."""
            task_q = _queue.Queue()
            for item in items:
                task_q.put(item)

            def pool_worker():
                while True:
                    try:
                        item = task_q.get_nowait()
                    except _queue.Empty:
                        break
                    i, link, attempt = item
                    entry = worker(i, link, attempt)
                    results_ordered[i] = entry
                    progress_q.put({"type": "result", "entry": entry})
                    task_q.task_done()

            threads = []
            for _ in range(MAX_CONCURRENT):
                t = _threading.Thread(target=pool_worker, daemon=True)
                t.start()
                threads.append(t)

            # Watchdog: сигнализирует когда все pool-воркеры завершились
            all_done_evt = _threading.Event()
            def _watchdog():
                for t in threads:
                    t.join()
                all_done_evt.set()
            _threading.Thread(target=_watchdog, daemon=True).start()

            done = 0
            total_batch = len(items)
            while done < total_batch:
                try:
                    msg = progress_q.get(timeout=3)  # не блокируем вечно
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg["type"] == "result":
                        done += 1
                except _queue.Empty:
                    if all_done_evt.is_set():
                        # Все воркеры закончили, но результатов меньше ожидаемого
                        # Предотвращаем вечное зависание
                        break
                    # Keepalive: SSE-комментарий чтобы прокси не закрыл соединение
                    yield ": keepalive\n\n"

        # ── Основной проход ──────────────────────────────────────────
        indexed_links = list(enumerate(links))
        _random.shuffle(indexed_links)
        batch = [(i, link, 1) for i, link in indexed_links]
        yield from run_batch(batch)

        # ── Retry для failed/timeout ─────────────────────────────────
        for attempt in range(2, MAX_RETRIES + 2):
            retry_list = [(i, links[i]) for i in range(total)
                          if results_ordered[i] and not results_ordered[i].get("ok")]
            if not retry_list:
                break
            yield f"data: {json.dumps({'type': 'retry_start', 'attempt': attempt, 'count': len(retry_list)})}\n\n"
            retry_batch = [(i, link, attempt) for i, link in retry_list]
            yield from run_batch(retry_batch)

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
