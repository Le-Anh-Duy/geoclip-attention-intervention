# Cách hoạt động của tool này — giải thích chi tiết

Tài liệu này giải thích **toàn bộ cách công cụ được xây dựng**: kiến trúc, luồng dữ liệu, và từng đoạn code quan trọng — kể cả các syntax Python/JS không quen thuộc. Viết cho người đã hiểu numpy và cơ chế attention cơ bản (Q, K, V, softmax).

---

## 1. Bức tranh tổng thể

```
┌─────────────────────┐         HTTP (multipart form)        ┌──────────────────────────┐
│   Frontend (React)  │ ────────────────────────────────────▶ │   Backend (FastAPI)      │
│  - vẽ vùng trên ảnh  │                                       │  - patch attention CLIP  │
│  - chỉnh a/b theo    │ ◀──────────────────────────────────── │  - chạy GeoCLIP 2 lần    │
│    từng layer        │        JSON (baseline + intervention) │    (baseline, intervened)│
│  - vẽ bản đồ kết quả │                                       │                          │
└─────────────────────┘                                       └──────────────────────────┘
```

Không có training, không có gradient. Đây thuần là **chạy forward pass 2 lần** — một lần bình thường (baseline), một lần với attention bị "nắn" theo vùng bạn chọn (intervention) — rồi so sánh kết quả.

Model GeoCLIP = CLIP ViT-L/14 (encode ảnh) + location encoder (encode toạ độ GPS thành vector, không liên quan gì đến ảnh) + một gallery ~100K điểm GPS. Dự đoán = tìm điểm GPS trong gallery có vector gần nhất với vector ảnh (theo cosine similarity → softmax ra xác suất).

**Điểm mấu chốt của cả dự án:** ta không đụng vào location encoder hay gallery — ta chỉ can thiệp vào **attention bên trong ViT khi nó encode ảnh**, để xem nó thay đổi kết quả dự đoán địa lý thế nào.

---

## 2. Công thức can thiệp (nhắc lại cho rõ)

Trong self-attention chuẩn, với mỗi query token, ta có:

```
attn_weights = softmax(Q · Kᵀ / √d)      # shape: (num_queries, num_keys)
output       = attn_weights · V
```

`attn_weights[i, j]` = query `i` "chú ý" bao nhiêu vào key `j`. Đây là một phân phối xác suất — mỗi hàng cộng lại bằng 1.

**Can thiệp** chèn thêm 1 bước, ngay sau softmax, ngay trước khi nhân với V:

```
scale[j] = a   nếu key j nằm trong vùng bạn chọn
scale[j] = b   nếu key j nằm ngoài vùng
attn_weights[i, j] *= scale[j]           # nhân theo cột (theo key)
attn_weights[i, :] /= sum(attn_weights[i, :])   # renormalize để hàng lại cộng = 1
```

Với `a > 1, b < 1`: mọi token (kể cả CLS) sẽ "hút" thông tin từ vùng bạn chọn nhiều hơn một chút, và từ phần còn lại ít hơn một chút. `a = b = 1` ⇒ scale toàn 1 ⇒ không đổi gì → **baseline chính là intervention với a=b=1**, dùng chung 1 hàm code.

CLS token (vị trí 0 trong sequence) — đại diện "toàn ảnh", không phải 1 patch không gian cụ thể — **không bao giờ bị scale khi đóng vai trò key** (luôn giữ scale=1), vì nó không có ý nghĩa "trong/ngoài vùng".

---

## 3. Backend — `backend/app/intervention.py`

Đây là file lõi, thực hiện đúng công thức trên. Đọc theo thứ tự nó chạy trong thực tế.

### 3.1. `InterventionState` — nơi giữ cấu hình hiện tại (dòng 26-50)

```python
@dataclass
class InterventionState:
    layer_ab: dict = field(default_factory=dict)          # {layer_idx: (a, b)}
    in_region_mask: torch.Tensor | None = None             # bool[256], patch nào trong vùng
```

`@dataclass` là syntax Python tự sinh `__init__` từ khai báo field — tương đương viết tay `def __init__(self, layer_ab={}, in_region_mask=None): ...` nhưng gọn hơn. `field(default_factory=dict)` là cách bắt buộc để default value là 1 dict *mới* mỗi lần khởi tạo object (nếu viết `layer_ab: dict = {}` trực tiếp, Python sẽ dùng **chung một dict** cho mọi instance — bug kinh điển).

`layer_ab` mặc định rỗng: layer nào không có trong dict thì coi như `(1, 1)` — không can thiệp.

```python
def key_scale_for_layer(self, layer_idx, num_positions):
    a, b = self.layer_ab.get(layer_idx, (1.0, 1.0))
    if a == 1.0 and b == 1.0:
        return None                          # no-op nhanh, khỏi tính toán thừa
    if self.in_region_mask is None:
        return None
    scale = torch.full((num_positions,), b, dtype=torch.float32)   # mặc định tất cả = b
    scale[0] = 1.0                            # CLS key: luôn giữ nguyên
    scale[1:] = torch.where(self.in_region_mask, torch.tensor(a), torch.tensor(b))
    return scale
```

Đây chính là hàm build ra vector `scale[j]` ở công thức trên, cho **một layer cụ thể**. `torch.where(cond, x, y)` = "nếu cond[i] đúng thì lấy x[i] không thì lấy y[i]" — element-wise, giống `np.where`. `num_positions` = 257 (1 CLS + 256 patch).

### 3.2. `_intervened_attention_forward` — nơi scale thật sự xảy ra (dòng 53-90)

Đây là hàm **factory** — một hàm trả về một hàm khác:

```python
def _intervened_attention_forward(state: InterventionState, layer_idx: int):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        ...
    return forward
```

Lý do cần pattern này: mỗi layer trong 24 layer của ViT cần một hàm `forward` *khác nhau* (khác `layer_idx` để tra đúng `(a,b)` của layer đó), nhưng logic bên trong giống hệt nhau. Gọi `_intervened_attention_forward(state, 5)` sẽ trả về một hàm `forward` "nhớ" `layer_idx=5` nhờ closure (biến `state`, `layer_idx` được hàm `forward` bên trong "đóng gói" theo, dù hàm ngoài đã return xong).

Bên trong `forward`, đây là **q/k/v projection + eager attention viết lại bằng tay**, y hệt logic gốc trong thư viện `transformers` (`CLIPAttention.forward`), *ngoại trừ* chèn thêm bước can thiệp:

```python
queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
keys    = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
values  = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale   # Q·Kᵀ / √d
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)

key_scale = state.key_scale_for_layer(layer_idx, attn_weights.shape[-1])
if key_scale is not None:
    attn_weights = attn_weights * key_scale.to(attn_weights.dtype)     # ← CAN THIỆP Ở ĐÂY
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)  # renormalize

attn_output = torch.matmul(attn_weights, values)     # attn_weights · V
```

`attn_weights` có shape `(batch, num_heads, num_queries, num_keys)`. `key_scale` shape `(num_keys,)` — khi nhân, PyTorch tự **broadcast** nó vào chiều cuối cùng (giống numpy broadcasting), tức là scale áp dụng theo **cột** (theo key), giống hệt mọi query đều nhân cùng 1 vector scale — đúng ý "mọi token đều hút vùng được chọn nhiều hơn", không riêng CLS.

**Vì sao phải viết lại tay** thay vì chỉnh trực tiếp: thư viện `transformers` hiện đại (v5.x) đóng gói attention qua `ALL_ATTENTION_FUNCTIONS` — một dispatch table (eager/sdpa/flash...) không có "khe hở" nào để chèn thêm bước ở giữa softmax và matmul V. Nên cách đơn giản nhất (ponytail: ít code nhất mà đúng) là copy lại đúng logic eager attention rồi chèn 3 dòng vào giữa, thay vì monkey-patch sâu vào nội bộ thư viện.

### 3.3. `patch_vision_tower` — gắn vào model (dòng 93-105)

```python
def patch_vision_tower(clip_model) -> InterventionState:
    state = InterventionState()
    layers = clip_model.vision_model.encoder.layers    # 24 CLIPEncoderLayer
    for layer_idx, layer in enumerate(layers):
        layer.self_attn.forward = types.MethodType(
            _intervened_attention_forward(state, layer_idx), layer.self_attn
        )
    return state
```

Chạy **một lần duy nhất** lúc server khởi động (xem `main.py` §4.1). `types.MethodType(fn, obj)` là cách Python "bind" một hàm rời rạc thành method của một object cụ thể — tức là sau dòng này, khi code gốc của CLIP gọi `layer.self_attn(...)`, nó sẽ chạy hàm `forward` mới của ta thay vì hàm gốc, với `self` tự động là đúng `self_attn` object đó (để vẫn truy cập được `self.q_proj`, `self.scale`, v.v. — các weight đã train sẵn, không đổi gì).

Trả về `state` — object **dùng chung** cho toàn bộ 24 layer. Mỗi request sau này chỉ cần *sửa* `state.layer_ab` / `state.in_region_mask` (không patch lại model) để đổi cấu hình can thiệp.

### 3.4. `build_in_region_mask` — map vùng bạn vẽ sang patch nào (dòng 118-154)

Bạn vẽ vùng theo % trên ảnh gốc (VD ảnh 1920×1080). Nhưng CLIP tự tiền xử lý ảnh trước khi chia patch: resize cạnh ngắn về 224px, rồi center-crop đúng 224×224. Nếu không tái tạo đúng phép biến đổi này, vùng bạn vẽ sẽ **lệch khỏi patch thực tế mà không báo lỗi gì** — sai âm thầm, nguy hiểm nhất trong loại bug này.

```python
scale = image_size / min(orig_w, orig_h)          # resize theo cạnh ngắn
resized_w, resized_h = orig_w * scale, orig_h * scale
crop_x0 = (resized_w - image_size) / 2             # center-crop offset
crop_y0 = (resized_h - image_size) / 2
```

Với mỗi vùng (toạ độ normalized 0-1 theo ảnh gốc), áp đúng phép scale + trừ crop-offset để ra toạ độ pixel trong khung 224×224 mà CLIP thực sự nhìn thấy, rồi cắt (`clip`) phần nằm ngoài khung crop, rồi chia cho `patch = 224/grid` (grid=16 → patch=14px) để ra chỉ số hàng/cột patch:

```python
col0, col1 = int(x0 // patch), min(int((x1 - 1e-6) // patch) + 1, grid)
row0, row1 = int(y0 // patch), min(int((y1 - 1e-6) // patch) + 1, grid)
mask[row0:row1, col0:col1] = True
```

`- 1e-6` trước khi chia là để tránh lỗi làm tròn khi `x1` rơi *đúng* trên biên patch (VD `x1 = 28.0` đúng biên giữa patch 1 và 2 — không trừ epsilon sẽ vô tình bao thêm 1 patch thừa). `mask.flatten()` cuối cùng biến `(16,16)` thành `(256,)` để khớp thứ tự patch token mà CLIP xuất ra (row-major, giống cách ViT patchify ảnh).

---

## 4. Backend — `backend/app/main.py`

### 4.1. Load model không chặn server (dòng 36-56)

```python
def _load_model_background():
    global _state
    model = get_model()                                        # tải weight CLIP từ HF Hub (lần đầu ~vài phút)
    _state = intervention.patch_vision_tower(model.image_encoder.CLIP)
    _set_status("ready")

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_model_background, daemon=True).start()
    yield
```

`lifespan` là hook chuẩn của FastAPI chạy quanh vòng đời app (code trước `yield` = lúc startup, sau `yield` = lúc shutdown). Thay vì tải model *trong* startup (sẽ chặn app không nhận request nào cho tới khi xong), ta bắn nó sang 1 thread nền (`daemon=True` = thread tự chết khi process chính chết, khỏi cần dọn) rồi `yield` ngay — app nhận request được ngay lập tức, `/health` trả về `"loading"` cho tới khi thread nền set `"ready"`.

### 4.2. Vì sao `/predict` phải là `def` chứ không phải `async def`

```python
@app.post("/predict", response_model=PredictResponse)
def predict(...):
```

FastAPI/Starlette chạy `async def` endpoint **thẳng trên event loop chính** — nếu bên trong có code CPU-bound đồng bộ (như `model.predict()`, chạy hàng chục giây trên CPU), nó chặn *toàn bộ* event loop, mọi request khác (kể cả `/health`) phải đợi. Với `def` (đồng bộ) thường, Starlette tự động chạy nó trong 1 thread pool riêng (`anyio` threadpool), giải phóng event loop để tiếp tục phục vụ request khác song song. Đây từng là bug thực tế đã gặp và sửa — verify bằng curl: gọi `/health` trong lúc `/predict` đang chạy vẫn trả về ngay (`{"state":"busy"}`).

### 4.3. Luồng `/predict` (dòng 109-162)

```python
image_bytes = image.file.read()
orig_w, orig_h = Image.open(io.BytesIO(image_bytes)).size

in_region_mask = intervention.build_in_region_mask(regions_parsed, orig_w, orig_h, grid)

with _predict_lock:
    _state.layer_ab = {}
    _state.in_region_mask = None
    baseline = _run(model, tmp_path, top_k, ground_truth_parsed)     # chạy KHÔNG can thiệp

    _state.layer_ab = layer_configs_parsed
    _state.in_region_mask = in_region_mask
    intervened = _run(model, tmp_path, top_k, ground_truth_parsed)   # chạy CÓ can thiệp

    _state.layer_ab = {}
    _state.in_region_mask = None      # dọn lại về trạng thái sạch cho request sau
```

Mỗi request: đọc kích thước ảnh gốc → build mask patch → **chạy model.predict() 2 lần**, lần 1 với `state` rỗng (baseline), lần 2 với `state` set theo config bạn gửi lên (intervention) — chung một model, chỉ khác giá trị trong `InterventionState` giữa 2 lần gọi, vì hàm `forward` đã patch đọc `state` **mỗi lần chạy** (không cache).

`_predict_lock` — một `threading.Lock` toàn cục — đảm bảo 2 request `/predict` đồng thời không "đá" lẫn config của nhau (vì `_state` là 1 object dùng chung, không phải per-request). Đây là simplification có chủ đích (ghi rõ trong code là "ponytail: single global lock... fine for a local single-user sandbox tool") — tool này chỉ dùng 1 người 1 lúc, nên đủ dùng; nếu sau này nhiều người dùng cùng lúc, cần đổi sang state per-request thay vì lock tuần tự.

---

## 5. Frontend

### 5.1. Vẽ vùng — `ImageRegionSelector.jsx`

Vùng được lưu dưới dạng **fraction 0-1 theo khung ảnh hiển thị** (`{x, y, w, h}`), không phải pixel. Vì thẻ `<img>` giữ nguyên tỉ lệ khung hình (`width:100%; height:auto`), fraction-theo-khung-hiển-thị == fraction-theo-ảnh-gốc — nên gửi thẳng lên backend, không cần quy đổi gì thêm, khớp đúng input mà `build_in_region_mask` (§3.4) mong đợi.

Kéo-thả: `onDragOver` phải gọi `e.preventDefault()` — mặc định trình duyệt sẽ *chặn* drop (mở file trong tab mới) nếu không preventDefault; đây là quy tắc bắt buộc của HTML5 Drag-and-Drop API, không phải lựa chọn thiết kế.

### 5.2. Chỉnh a/b — `LayerControls.jsx`

Mỗi layer là 1 hàng, gồm 2 cặp slider+number cho `a`/`b`. State ở dạng `{layerIdx: [a, b]}`, gửi thẳng lên backend qua `layer_configs` — khớp với `layer_ab` mà `InterventionState` mong đợi (dict `int → (a,b)`), chỉ khác JSON không có kiểu tuple nên backend tự `tuple(v)` lại khi parse (xem `main.py` dòng 126).

### 5.3. Kết quả + bản đồ — `ResultsPanel.jsx`

Dùng Leaflet.js thuần (không qua react-leaflet, đỡ 1 dependency). Điểm hay ho về React ở đây là **callback ref** thay vì `useRef` + `useEffect(..., [])`:

```jsx
function attachMap(node) {
  if (!node || mapInstance.current) return
  mapInstance.current = L.map(node).setView([20, 0], 2)
  ...
}
...
<div ref={attachMap} className="map" />
```

Component này `return null` (không render gì, kể cả div bản đồ) cho tới khi có `result` đầu tiên. Nếu dùng `useRef` + `useEffect(() => {...}, [])` (chạy đúng 1 lần lúc mount), effect đó sẽ chạy ngay ở lần mount đầu tiên — lúc mà `<div className="map">` **còn chưa tồn tại trong DOM** (vì `return null` phía trên) — nên `mapRef.current` mãi mãi là `null`, bản đồ không bao giờ khởi tạo. Đây chính là bug đã gặp và sửa. Callback ref (`ref={attachMap}`) thì khác: React tự gọi nó **mỗi khi node thực sự gắn vào DOM**, bất kể lúc đó là lần render nào — nên dù div chỉ xuất hiện ở lần render thứ N (khi `result` khác `null`), `attachMap` vẫn được gọi đúng lúc đó.

Marker độ mờ/kích thước giảm dần theo rank (`markerStyle`) để bản đồ đọc như "đám mây độ tin cậy" thay vì danh sách chấm phẳng.

### 5.4. Tên địa danh — `geocode.js`

Gọi Nominatim (OpenStreetMap, miễn phí, không cần API key — cùng nguồn với map tile nên không thêm dependency mới). Giới hạn dùng free ~1 request/giây, nên có **hàng đợi tuần tự**:

```js
let queue = Promise.resolve()
export function reverseGeocode(lat, lon) {
  const key = cacheKey(lat, lon)
  if (cache.has(key)) return cache.get(key)
  const promise = queue.then(() => fetchPlaceName(lat, lon))
  queue = promise                    // request tiếp theo phải đợi request này xong
  cache.set(key, promise)
  return promise
}
```

Mỗi lần gọi, promise mới được **nối đuôi** vào `queue` hiện tại (`queue.then(...)`) rồi `queue` được cập nhật lại thành chính nó — nên dù gọi `reverseGeocode` 10 lần liên tiếp (10 marker trên bản đồ), chúng chạy **tuần tự** chứ không song song, cách nhau tối thiểu 1100ms (`finally` trong `fetchPlaceName` — chạy dù thành công hay lỗi). Cache theo toạ độ làm tròn 3 chữ số thập phân (~110m) để tránh gọi lại API cho cùng 1 điểm.

### 5.5. Trạng thái server — `StatusBadge.jsx`

Poll `GET /health` mỗi 2s. `predicting` prop (từ `App.jsx`, bật true ngay khi bấm nút chạy) override hiển thị thành "busy" **ngay lập tức**, không cần đợi tick poll tiếp theo — vì `_predict_lock.locked()` phía backend chỉ *bắt đầu* true từ lúc `/predict` handler thật sự chạy tới dòng `with _predict_lock:`, có độ trễ nhỏ so với lúc frontend bấm nút.

---

## 6. Luồng đầy đủ 1 lần chạy

1. Bạn upload ảnh, vẽ 1-2 vùng, chỉnh `a`/`b` cho vài layer, nhập ground truth (tuỳ chọn), bấm "Chạy inference".
2. Frontend gửi `POST /predict` (multipart form: ảnh + JSON regions + JSON layer_configs + ground_truth + top_k).
3. Backend: parse JSON, đọc kích thước ảnh gốc, build `in_region_mask` (16×16 bool) bằng cách tái tạo phép resize+crop của CLIP.
4. Chạy `model.predict()` lần 1 với `state` rỗng → **baseline**.
5. Set `state.layer_ab`/`in_region_mask` theo config bạn gửi → chạy `model.predict()` lần 2 → mỗi lần 1 trong 24 layer tự động dùng `forward` đã patch, tự tra `state` để biết có scale hay không (layer không có trong `layer_configs` thì `key_scale_for_layer` trả `None`, chạy y hệt attention gốc) → **intervention**.
6. Trả cả 2 kết quả (top-k GPS + xác suất + khoảng cách tới ground truth nếu có) về frontend.
7. Frontend vẽ 2 cột kết quả, vẽ bản đồ với marker của cả baseline (xanh) và intervention (đỏ) và ground truth (nếu có), tra tên địa danh qua Nominatim, cho phép lưu lại để so sánh nhiều lần chạy.

---

## 7. Các giới hạn/đơn giản hoá có chủ đích (ponytail)

- **1 global lock** thay vì state per-request — đủ cho 1 người dùng, sẽ cần đổi nếu nhiều người dùng đồng thời.
- **Không auth, không HTTPS** — chỉ chạy local (`127.0.0.1` ↔ `localhost:5173`), CORS giới hạn đúng origin đó.
- **Chỉ can thiệp vision tower**, không đụng location encoder / gallery — đúng scope nghiên cứu đã chốt.
- **CPU-only**, có cap `torch.set_num_threads(8)` để tránh oversubscription trên máy nhiều core — tốc độ vẫn phụ thuộc nhiều vào RAM trống của máy lúc chạy (đã quan sát: máy ít RAM trống → chậm hẳn, không phải do code).
