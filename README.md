# Yuz va tana tanib olish tizimi

Kamera orqali odamlarni **yuzi** va **tanasi** (gavda, kiyim) bo'yicha taniydigan, har safar yangi
ko'rinishlarni o'rganib boradigan tizim. Barcha ma'lumotlar `data/database.db` (SQLite) bazasida saqlanadi.

## Ishga tushirish

```bash
./run.command                  # yoki: .venv/bin/python server.py
```

Brauzerda avtomatik ochiladi: **http://127.0.0.1:8765** (Chrome, Safari, Edge). Kameraga ruxsatni brauzer so'raydi.

```bash
.venv/bin/python server.py --port 9000 --no-browser
```

Birinchi marta (yangi kompyuterda):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py
```

Interfeys: `web/` (HTML/CSS/JS), server va API: `server.py`, tanish dvigateli: `recognizer.py`.

## Bo'limlar

| Bo'lim | Vazifasi |
|---|---|
| **Jonli kamera** | Real vaqtda tanish. Yashil – tanilgan, qizil – noma'lum, ko'k – faqat tana orqali. "Hozir kadrda" ro'yxati, faollik jurnali, rasmdan aniqlash. |
| **Ro'yxatdan o'tkazish** | Anketa (F.I.Sh, tug'ilgan sana, telefon, email, bo'lim, lavozim, ID, izoh). So'ng yuz kamera orqali (6 qadam, 24 namuna) yoki rasmlardan olinadi. |
| **Foydalanuvchilar** | Qidirish, profil kartasi, tahrirlash, o'chirish, kameradan/rasmlardan qo'shimcha o'rgatish, o'rganilgan namunalarni ko'rish. |
| **Sozlamalar** | Tanish/o'rganish chegaralarini slayder bilan sozlash. |
| **Dashboard** | Ko'rsatkichlar (jami, bugun kelganlar, aniqlashlar, noma'lumlar, namunalar), 14 kunlik va soatlik grafiklar, hodisalar tarixi (surati bilan), faollik reytingi, CSV eksport. |

## Qanday o'rganadi

1. Ro'yxatdan o'tishda turli burchak va ifodalardagi 24 ta yuz namunasi + tana namunalari olinadi.
2. Keyin kamera odamni **ishonchli** tanisa (≥5 kadr ketma-ket, o'xshashlik ≥0.55, boshqa odam bilan
   adashish xavfi yo'q, rasm tiniq), va bu ko'rinish bazadagi namunalardan **farq qilsa** (boshqa yorug'lik,
   burchak, soch, ko'zoynak, soqol...), u yangi namuna sifatida saqlanadi. Tana namunalari ham shu paytda
   yangilanadi (kiyim har kuni o'zgaradi).
3. "Rasmlardan o'rgatish" orqali har xil turdagi rasmlar (telefon, hujjat, guruh surati, oq-qora) qo'shiladi;
   guruh suratida shu odamga eng o'xshash yuz tanlanadi.
4. Galereya to'lsa (yuz – 150, tana – 60), yuz uchun eng takroriy namuna, tana uchun eng eskisi o'chiriladi.
5. Yuz chegaraviy o'xshash bo'lsa, tana ham tasdiqlasa – qabul qilinadi; yuz ko'rinmasa (orqasi bilan,
   uzoqda) – faqat tana orqali taniladi.

## Ma'lumotlar bazasi (`data/database.db`)

| Jadval | Mazmuni |
|---|---|
| `users` | Anketa ma'lumotlari, profil surati, yaratilgan/yangilangan vaqt |
| `embeddings` | Yuz (128) va tana (768) vektorlari, manbasi (`reg` / `auto` / `photo`) |
| `samples` | O'rganilgan yuz rasmlari (har bir odamga ≤200 ta) |
| `events` | Hodisalar: ro'yxatdan o'tdi, aniqlandi, noma'lum, o'rgandi, tahrirlandi, o'chirildi (kadr surati bilan) |

Bazani istalgan SQLite dasturida (masalan, DB Browser for SQLite) ochish mumkin.

## Sozlamalar (`settings.json`)

Birinchi ishga tushishda yaratiladi. Asosiylari:

* `face_match` (0.42) – yuz tanish chegarasi. Begonalar tanilib qolsa – oshiring, o'zingizni tanimasa – kamaytiring.
* `face_learn` (0.55) – avtomatik o'rganish uchun minimal ishonch.
* `face_novelty` (0.80) – yangi namuna deb hisoblash chegarasi (kichikroq = kamroq, lekin xilma-xilroq namuna).
* `body_match` (0.72) – tana orqali tanish chegarasi.

## Texnologiyalar

OpenCV DNN + ONNX modellari (OpenCV Zoo): **YuNet** (yuz detektori), **SFace** (yuz embedding),
**Youtu ReID** (tana embedding), **HOG** (odam detektori). GPU va internet talab qilinmaydi.

> Eslatma: biometrik ma'lumotlarni yig'ishdan oldin odamlarning roziligini oling.

## Hisoblar va rollar

| Rol | Ko'radigan bo'limlar |
|---|---|
| **Administrator** | Hammasi + "Hisoblar" (rol berish, bloklash, parolni tiklash, o'chirish) |
| **Direktor** | Faqat Dashboard |
| **Foydalanuvchi** | Faqat jonli kamera |

* **Sign up** orqali yaratilgan hisob "Foydalanuvchi" rolini oladi; direktor/admin rolini faqat administrator beradi.
* Tizimdagi **birinchi** hisob avtomatik administrator bo'ladi.
* Ruxsatlar serverdagi har bir API'da tekshiriladi; parollar xeshlangan holda (`accounts` jadvali) saqlanadi.
* 5 marta noto'g'ri parol kiritilsa, 5 daqiqaga bloklanadi. Oxirgi administratorni o'chirib yoki bloklab bo'lmaydi.
* Boshqa kompyuterlardan kirish: `server.py --host 0.0.0.0`. Eslatma: brauzerlar kamerani faqat `localhost` yoki HTTPS orqali beradi.

## Netlify'ga joylash

Netlify faqat statik sayt va kichik JS funksiyalarni ishga tushiradi — Python, OpenCV, 300 MB modellar va doimiy
SQLite baza u yerda ishlamaydi. Shuning uchun loyiha ikki qismga bo'lingan:

```
Brauzer ──HTTPS──> Netlify (web/ : interfeys)
                      └── /api/*  ──proksi──> Backend (Docker: server.py + modellar + baza)
```

Foydalanuvchi uchun hammasi bitta domen: login cookie'lari ishlaydi, kamera HTTPS orqali ruxsat oladi.

### 1. Backend (Docker)

`Dockerfile` istalgan Docker hostingda ishlaydi (Render, Railway, Fly.io, VPS). Talablar:
**kamida 2 GB RAM** va `/data` ga ulangan **doimiy disk** (aks holda har qayta ishga tushganda baza o'chadi).

* **Render**: New → Blueprint → repozitoriy (`render.yaml` tayyor), yoki New → Web Service → Docker.
* **VPS**: `docker build -t faceid . && docker run -d -p 8080:8080 -v faceid-data:/data -e SECRET_KEY=... faceid`
  (oldiga HTTPS beruvchi Caddy/Nginx qo'ying).

Muhit o'zgaruvchilari: `SECRET_KEY` (uzun tasodifiy satr), `PROXY_HOPS` (proksilar soni, odatda 2).
Tekshirish: `https://<backend>/healthz` → `{"ok": true}`.

### 2. Frontend (Netlify)

1. Netlify → Add new site → Import from Git → shu repozitoriy (`netlify.toml` avtomatik o'qiladi).
2. Site configuration → Environment variables → `BACKEND_URL = https://<backend-manzil>`.
3. Deploy. `netlify/build.sh` `web/` ni `dist/` ga ko'chiradi va `/api/*` proksi qoidasini yozadi.

`BACKEND_URL` o'zgarsa — Netlify'da qayta deploy qiling.

### Git

`data/` (biometrik baza, parol xeshlari, sessiya kaliti) va `models/` `.gitignore` da — ularni **hech qachon**
git'ga qo'ymang. Modellar Docker build paytida `download_models.py` orqali yuklanadi.
