# Argon 4 আর Hydrogen — API আর Website গাইড

এই গাইডটা Gaymini Claude Pro সার্ভারের জন্য। সার্ভার চালু থাকলে এই দুই মডেল ব্যবহার করা যায়।

## ২টা মডেল

| মডেল | কিসের জন্য | গতি |
|---|---|---|
| `gemini-4-argon` | কঠিন লজিক, অংক, বড় কোড, গভীর চিন্তা | ধীর (৩০-৬০ সেকেন্ড) |
| `gemini-4-hydrogen` | রোজকার কোডিং, তাড়াতাড়ি উত্তর | ফাস্ট (১০-৩০ সেকেন্ড) |

দুটোই আসলে তোমার Gemini অ্যাকাউন্ট দিয়ে চলে। আলাদা কোনো টাকা লাগে না।

## ১. সার্ভার চালানো

```bash
cd gaymini-claude
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp তোমার-cookies.json cookies.json   # cookies.json ফাইলে কুকি বসাতে হয়
.venv/bin/python gemini_api_server.py
```

চালু হলে দেখবে: `Serving with waitress on http://127.0.0.1:5000`

চেক করো:

```bash
curl http://127.0.0.1:5000/health
```

`cookies_configured: true` আসলেই ঠিক আছে।

## ২. Website — চ্যাট পেজ

সার্ভারের নিজের পেজ আছে:

```
http://127.0.0.1:5000/chat
```

(`/` খুললে এখানেই পাঠিয়ে দেয়।) ওখানে Argon / Hydrogen বোতাম আছে, লিখে Send করলেই উত্তর আসে।

আরেকটা সহজ পেজ বানিয়ে দেওয়া হয়েছে: `chat.html`
এই ফাইলটা ব্রাউজারে খুললেই চলবে। ওপরে API ঠিকানা বসানো যায়,
মডেল বেছে নেওয়া যায়, আগের কথা মনে রাখে (history পাঠায়)।

চ্যাট পেজে ৩টা মোড আছে:
- **💬 চ্যাট** — Argon / Hydrogen দিয়ে কথা বলা। উত্তর মার্কডাউন আকারে দেখায়, কপি বোতাম আছে, আগের চ্যাট রিলোডের পরেও থাকে।
- **🎨 ছবি বানাও** — Nano Banana Pro দিয়ে নতুন ছবি বানানো (নিচে ৭ নম্বর দেখো)।
- **✏️ ছবি এডিট** — 📷 দিয়ে ছবি দাও, লিখো কী বদলাতে চাও। Azad AI বা Google — দুইটা দিয়ে এডিট করা যায় (নিচে ৭ নম্বর দেখো)।

## ২.৫ API Key

সার্ভারে API Key চালু আছে। `/health`, `/`, `/chat` খোলা, বাকি সব `/v1/*` আর ছবির রাস্তায় `x-api-key` হেডার লাগে (না হলে 401 আসে):

```bash
-H "x-api-key: তোমার-কী"
```

Key-টা `~/workspace/gaymini-claude/.api_key` ফাইলে আছে। চ্যাট পেজে Key আগেই বসানো থাকে, তাই ওখানে কিছু করতে হয় না।

## ৩. API — OpenAI স্টাইল (সহজটা, এটাই ব্যবহার করো)

### Hydrogen দিয়ে চ্যাট

```bash
curl -X POST http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: তোমার-কী" \
  -d '{"model": "gemini-4-hydrogen", "messages": [{"role": "user", "content": "2+2 koto?"}]}'
```

### Argon দিয়ে চ্যাট

```bash
curl -X POST http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: তোমার-কী" \
  -d '{"model": "gemini-4-argon", "messages": [{"role": "user", "content": "পাইথনে quicksort লিখে বুঝিয়ে দাও"}]}'
```

### Python কোড

```python
import requests

API = "http://127.0.0.1:5000"

def chat(message, model="gemini-4-hydrogen"):
    r = requests.post(
        f"{API}/v1/chat/completions",
        headers={"x-api-key": "তোমার-কী"},
        json={"model": model, "messages": [{"role": "user", "content": message}]},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

print(chat("বাংলায় একটা ছোট গল্প লেখো", model="gemini-4-argon"))
print(chat("2+2 koto?", model="gemini-4-hydrogen"))
```

### JavaScript (ওয়েবসাইটে ব্যবহারের জন্য)

```javascript
async function chat(message, model) {
  const res = await fetch("http://127.0.0.1:5000/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-api-key": "তোমার-কী" },
    body: JSON.stringify({
      model: model, // "gemini-4-argon" বা "gemini-4-hydrogen"
      messages: [{ role: "user", content: message }]
    })
  });
  const data = await res.json();
  return data.choices[0].message.content;
}
```

## ৪. API — Anthropic / Claude Code স্টাইল

Claude Code এই রাস্তা দিয়ে চালানো যায়:

```bash
curl -X POST http://127.0.0.1:5000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: তোমার-কী" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model": "gemini-4-hydrogen", "max_tokens": 500,
       "messages": [{"role": "user", "content": "Hi, tumi ke?"}]}'
```

Claude Code চালাতে:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:5000"
export ANTHROPIC_API_KEY="তোমার-কী"
export ANTHROPIC_DEFAULT_OPUS_MODEL="gemini-4-argon"
export ANTHROPIC_DEFAULT_SONNET_MODEL="gemini-4-hydrogen"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="gemini-4-hydrogen"
claude --model gemini-4-hydrogen
```

## ৫. মডেলের লিস্ট দেখা

```bash
curl http://127.0.0.1:5000/v1/models -H "x-api-key: তোমার-কী"
```

`nano-banana-pro` ও লিস্টে আছে — এটা চ্যাটের মডেল না, ছবির মডেল।

## ৭. ছবি বানানো আর এডিট (Nano Banana Pro / Google / Azad)

ছবির কাজ ১-৪ মিনিট লাগে, তাই সব async job: প্রথমে POST করলে একটা job id আসে, তারপর `GET /edit-status/<job>` দিয়ে ৪-৫ সেকেন্ড পর পর চেক করতে হয়। `status` `done` হলে `image_url` আসে।

### Nano Banana Pro দিয়ে নতুন ছবি বানাও

```bash
curl -X POST http://127.0.0.1:5000/generate-image \
  -H "x-api-key: তোমার-কী" \
  -F "prompt=a cute orange cat on a rooftop at sunset, anime style" \
  -F "origin=http://127.0.0.1:5000"
```

### Azad AI দিয়ে ছবি এডিট করো

```bash
curl -X POST http://127.0.0.1:5000/edit-image \
  -H "x-api-key: তোমার-কী" \
  -F "image=@ছবি.jpg" -F "prompt=make it sunset" \
  -F "origin=http://127.0.0.1:5000"
```

### Google দিয়ে ছবি এডিট করো

```bash
curl -X POST http://127.0.0.1:5000/edit-image-google \
  -H "x-api-key: তোমার-কী" \
  -F "image=@ছবি.jpg" -F "prompt=make it sunset" \
  -F "origin=http://127.0.0.1:5000"
```

### Job-এর অবস্থা দেখো

```bash
curl http://127.0.0.1:5000/edit-status/<job-id> -H "x-api-key: তোমার-কী"
```

### মনে রাখার কথা (ছবি)

- এডিট করা ছবি প্রথমে tmpfiles.org-এ রাখা হয় (ফোনে তাড়াতাড়ি ডাউনলোড হয়)। tmpfiles না পৌঁছালে সার্ভার নিজের লিংক (`/uploads/...`) দেয় — একটু ধীরে ডাউনলোড হবে, কিন্তু কাজ হবে।
- Nano Banana Pro আর Google এডিট — দুটোই তোমার Gemini সেশন দিয়ে চলে, আলাদা কী লাগে না। **২০২৬-১১-১৮ সকালে নতুন কুকি বসানোর পর ছবি বানানো আসলেই কাজ করেছে** (Naruto ছবি ৫৭ সেকেন্ডে বানিয়েছে) — আগের কুকি পুরনো হয়ে যাওয়ায় Google "signed out" ভেবে ছবি বানাতে দিচ্ছিল না। Google আবার "ছবি বানানো যায়নি" বললে প্রথমে নতুন কুকি বসিয়ে দেখো। আর Google ছবি না বানিয়ে খুঁজে পাওয়া ছবি দিলে সার্ভার এখন সেটাই দেখায় ("Google-এ পাওয়া ছবি" লেখা সহ)।
- ছবি এডিট ধীরে হলে: ফোনের ছবি পাঠানোর আগেই পেজে ছোট করে নেওয়া হয় (সবচেয়ে বড় দিক ১২৮০px), আর সার্ভার tmpfiles-এ একবারই চেষ্টা করে (৩০ সেকেন্ড) তারপর নিজের লিংক দেয় — ২০২৬-১১-১৮ টেস্টে পুরো এডিট ৫৩ সেকেন্ডে শেষ হয়েছে।

## ৬. মনে রাখার কথা

- উত্তর আসতে ১০-৪০ সেকেন্ড লাগতে পারে, এটা স্বাভাবিক।
- কুকি পুরনো হয়ে গেলে `authentication_error` আসবে — তখন নতুন কুকি বসিয়ে সার্ভার রিস্টার্ট করতে হবে।
- `cookies.json` কাউকে দেবে না, GitHub-এ তুলবে না। এটা তোমার Google অ্যাকাউন্টের চাবি।
- এই সার্ভার শুধু তোমার কম্পিউটারে (`127.0.0.1`) চলে। অন্য ফোন/কম্পিউটার থেকে ব্যবহার করতে হলে সার্ভারটা সেই কম্পিউটারেই চালাতে হবে, অথবা Codespace/VPS-এ চালিয়ে তার ঠিকানা ব্যবহার করতে হবে।
- একসাথে বেশি রিকোয়েস্ট পাঠালে Google 429 (ব্লক) দিতে পারে। একটা শেষ হলে আরেকটা পাঠাও।
- পাবলিক লিংক (localhost.run) কিছুক্ষণ পর পর নিজে থেকে বদলে যায় — নতুন লিংক `/tmp/localhost-run.log` থেকে দেখতে হয়, আর `chat.html`-এর ফলব্যাক ঠিকানাও বদলাতে হয়। সার্ভার বা কম্পিউটার রিস্টার্ট হলে লিংক মরে যায়।

## ফাইলগুলো

- `gemini_api_server.py` — আসল সার্ভার (একটা ভুল ঠিক করা হয়েছে, দেখো নিচে)
- `chat.html` — বাংলায় চ্যাট ওয়েবসাইট
- `app.py` — ছবি/ভিডিও/মিউজিক API-ও আছে (`/api/chat`, `/api/image`, `/api/video`, `/api/music`)
- `cookies.json` — তোমার কুকি (এই ফাইলটা zip-এ নেই, আলাদা করে বসাতে হবে)

## ঠিক করা ভুলটা

আগের `gemini_api_server.py`-তে ছিল `GeminiClient(session=session)` — কিন্তু
`GeminiClient`-এর `session` নামে কোনো প্যারামিটার নেই, তাই কুকি পৌঁছাতো না
আর লগে `Failed to initialize client after 0 attempts` আসতো।
এখন করা হয়েছে `GeminiClient(secure_1psid=psid, secure_1psidts=psidts)` —
যেভাবে `app.py`-তে করা আছে। এটা ঠিক করার পরেই লাইভ টেস্ট পাস করেছে।

## ৮. Google Flow ও Veo ৩.১ দিয়ে ভিডিও বানানো

Google Flow (`flow.google.com`) এবং Google Veo ৩.১ দিয়ে সরাসরি ভিডিও জেনারেট করা যায়। এটি টেক্সট থেকে ভিডিও (Text-to-Video) এবং ছবি থেকে ভিডিও (Image-to-Video) দুটোই সমর্থন করে।

### ক. টেক্সট থেকে ভিডিও বানাও

```bash
curl -X POST http://127.0.0.1:5000/generate-video \
  -H "x-api-key: তোমার-কী" \
  -F "prompt=cinematic drone shot of sunset over Dhaka city, 4k hyperrealistic" \
  -F "origin=http://127.0.0.1:5000"
```

### খ. ছবি অ্যানিমেট করে ভিডিও বানাও (Image-to-Video)

```bash
curl -X POST http://127.0.0.1:5000/generate-video \
  -H "x-api-key: তোমার-কী" \
  -F "image=@photo.jpg" \
  -F "prompt=animate this image: cinematic camera slow zoom, gentle breeze moving hair" \
  -F "origin=http://127.0.0.1:5000"
```

### গ. ভিডিওর অবস্থা ও প্রগ্রেস দেখো

```bash
curl http://127.0.0.1:5000/video-status/<job-id> -H "x-api-key: তোমার-কী"
```

ভিডিও রেডি হলে `status: "done"` এবং `video_url` আসবে, যা ব্রাউজারে সরাসরি চলবে এবং ফোনে `.mp4` হিসেবে ডাউনলোড করা যাবে।

## ৯. Google ও Google Flow অটো-লগইন ও সিঙ্ক সিস্টেম (Automatic Login)

বার বার কুকি কপি-পেস্ট করার ঝামেলা এড়াতে স্বয়ংক্রিয় অটো-লগইন ও সিঙ্ক আর্কিটেকচার যুক্ত করা হয়েছে:

1. **১-ট্যাপ অটো-সিঙ্ক বুকমার্কলেট (ফোন ও পিসি):**
   `chat.html`-এর উপরে **🔑** বোতাম চাপলে একটি বুকমার্কলেট পাওয়া যায়। Chrome-এ `flow.google.com` বা `gemini.google.com` খুলে বুকমার্কে এক ট্যাপ করলেই ব্রাউজারের গুগল ও ফ্লো সেশন স্বয়ংক্রিয়ভাবে সার্ভারে সিঙ্ক হয়ে যাবে।
2. **অটো-রোটেশন ব্যাকগ্রাউন্ড কিপ-অ্যালাইভ:**
   `__Secure-1PSIDTS` এবং সেশন মেয়াদোত্তীর্ণ হওয়া রোধ করতে ব্যাকগ্রাউন্ডে Google `RotateCookies` কল হয় এবং নতুন কুকি স্বয়ংক্রিয়ভাবে `cookies.json`-এ আপডেট হয়।
3. **সরাসরি সিঙ্ক API:**
   ```bash
   curl -X POST http://127.0.0.1:5000/api/auth/sync \
     -H "Content-Type: application/json" \
     -d '{"cookies": "SAPISID=...; OSID=...; __Secure-1PSID=..."}'
   ```
   যেকোনো ফরম্যাটের কুকি (JSON, Header String) স্বয়ংক্রিয়ভাবে ফিল্টার ও সেভ হয়ে যাবে।

## ১০. একাধিক একাউন্ট ভল্ট (Multiple Accounts System) ও ফোন এক্সেস

### ক. ফোনে সরাসরি চালানো ও ব্যবহার (Phone Access)

সার্ভার এখন `0.0.0.0:5000`-এ চলে, তাই একই ওয়াইফাই বা হটস্পটে থাকা যেকোনো মোবাইল ফোনের ব্রাউজারে সরাসরি খুলুন:
```
http://192.168.0.112:5000/chat
```

### খ. একাধিক গুগল একাউন্ট যোগ ও ১-ক্লিকে সুইচ (Multiple Accounts)

- **সব একাউন্ট দেখা:**
  `GET /api/accounts`
- **একাউন্ট পরিবর্তন (১-ক্লিক সুইচ):**
  `POST /api/accounts/switch` with `{"id": "account_id"}`
- **নতুন একাউন্ট যোগ:**
  `POST /api/accounts/add` with `{"name": "Pro Account 2", "cookies": "..."}`
- **একাউন্ট ডিলিট:**
  `POST /api/accounts/delete` with `{"id": "account_id"}`

`chat.html`-এর উপরে **🔑** বোতাম খুললে সব একাউন্টের তালিকা, স্ট্যাটাস ব্যাজ ও সুইচ বোতাম পাওয়া যায়।

### গ. ন্যানো বানানা প্রো (Nano Banana Pro) ফ্রি মডেল

চ্যাট পেজে **🎨 ছবি বানাও** মোডে দুটি অপশন যোগ করা হয়েছে:
1. `🍌 Nano Banana Pro (Free)`: কোনো কুকি বা কোটা ঝামেলা ছাড়াই আনলিমিটেড ফ্রি হাই-কোয়ালিটি ছবি তৈরি করে।
2. `🎨 Google Flow (Imagen 3)`: গুগল ফ্লো-এর হাই-রেজোলিউশন মডেল ব্যবহার করে।

### ঘ. সিকিউরিটি নোট (গুগল প্রো একাউন্টের নিরাপত্তা)

গুগল একাউন্টের ইউজারনেম ও পাসওয়ার্ড দিয়ে কোনো স্বয়ংক্রিয় স্ক্রিপ্ট দিয়ে লগইন করালে গুগল বট ডিটেকশন ও ২-ফ্যাক্টর ব্লকিং সক্রিয় হয়ে প্রো একাউন্ট পার্মানেন্টলি ব্যান হয়ে যেতে পারে। এজন্য আমাদের সিস্টেম পাসওয়ার্ড শেয়ার করার কোনো ঝুঁকি নেয় না। পরিবর্তে, ব্যাকগ্রাউন্ডে অফিশিয়াল Google `RotateCookies` ডেমন প্রতি ১২ মিনিটে সেশন সতেজ রাখে এবং `accounts_vault.json` কঠোর `chmod 600` পারমিশনে সংরক্ষিত থাকে — ফলে একাউন্ট থাকবে ১০০% নিরাপদ।

