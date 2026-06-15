# ── Patch gradio_client BEFORE importing gradio ──────────────────
import gradio_client.utils as _gcu
_orig_get_type = _gcu.get_type
def _patched_get_type(schema):
    if not isinstance(schema, dict): return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _patched_get_type
_orig_js = _gcu.json_schema_to_python_type
def _safe_js(schema, defs=None):
    try: return _orig_js(schema, defs)
    except: return "Any"
_gcu.json_schema_to_python_type = _safe_js

try:
    import pillow_heif; pillow_heif.register_heif_opener()
except ImportError: pass

import json, traceback, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocabulary:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3
    def __init__(self, d):
        self.stoi = d["stoi"]
        self.itos = {int(k): v for k, v in d["itos"].items()}
    def __len__(self): return len(self.stoi)
    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.EOS: break
            if i not in (self.PAD, self.SOS): out.append(self.itos.get(i, "?"))
        return " ".join(out)

class EfficientNetEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.features = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
        self.proj = nn.Sequential(nn.Linear(1280, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim) * 0.02)
        for p in self.features.parameters(): p.requires_grad = False
    def set_fine_tune(self, enable=True):
        for p in self.features.parameters(): p.requires_grad = False
        if enable:
            for i in [6, 7]:
                for p in self.features[i].parameters(): p.requires_grad = True
    def forward(self, x):
        return self.proj(self.features(x).flatten(2).permute(0, 2, 1)) + self.pos_embed

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=6, ff_dim=1024, max_len=52, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos   = nn.Embedding(max_len, embed_dim)
        self.drop  = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerDecoder(layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.head.weight = self.embed.weight
    def forward(self, t, mem):
        T = t.size(1)
        pos = torch.arange(T, device=t.device).unsqueeze(0)
        x = self.drop(self.embed(t) + self.pos(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=t.device)
        return self.head(self.norm(self.transformer(x, mem, tgt_mask=mask)))

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=6, ff_dim=1024, max_len=52, dropout=0.1):
        super().__init__()
        self.encoder = EfficientNetEncoder(embed_dim)
        self.decoder = TransformerDecoder(vocab_size, embed_dim, num_heads,
                                          num_layers, ff_dim, max_len, dropout)
    @torch.no_grad()
    def generate_greedy(self, t, max_len=30):
        self.eval()
        mem = self.encoder(t.unsqueeze(0).to(DEVICE))
        tok = [Vocabulary.SOS]
        for _ in range(max_len):
            nid = self.decoder(torch.tensor([tok], device=DEVICE), mem)[0, -1].argmax().item()
            if nid == Vocabulary.EOS: break
            tok.append(nid)
        return vocab.decode(tok[1:])
    @torch.no_grad()
    def generate_beam(self, t, k=5, max_len=30):
        self.eval()
        mem = self.encoder(t.unsqueeze(0).to(DEVICE))
        beams, done = [(0.0, [Vocabulary.SOS])], []
        for _ in range(max_len):
            cands = []
            for sc, sq in beams:
                if sq[-1] == Vocabulary.EOS: done.append((sc, sq)); continue
                lp = F.log_softmax(
                    self.decoder(torch.tensor([sq], device=DEVICE), mem)[0, -1], dim=-1)
                for v, i in zip(*lp.topk(k)):
                    cands.append((sc + v.item(), sq + [i.item()]))
            if not cands: break
            cands.sort(key=lambda x: x[0]/len(x[1]), reverse=True)
            beams = cands[:k]
        best = max(done + beams, key=lambda x: x[0]/max(len(x[1]), 1))
        return vocab.decode(best[1][1:])

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════
print("Loading vocabulary...")
with open("vocab.json") as f: vocab = Vocabulary(json.load(f))

def load_custom(path, label):
    m = ImageCaptioningModel(vocab_size=len(vocab)).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    m.load_state_dict(ckpt["model"]); m.eval()
    print(f"  [{label}] val_loss={ckpt.get('val_loss','?')}")
    return m

model_5k = None
if os.path.exists("best_phase1.pt"):
    try: model_5k = load_custom("best_phase1.pt", "5k")
    except Exception as e: print(f"  5k: {e}")

model_100k = None
if os.path.exists("best_phase2.pt"):
    try: model_100k = load_custom("best_phase2.pt", "100k")
    except Exception as e: print(f"  100k: {e}")

blip_model, blip_processor = None, None
if os.path.exists("blip_adapter"):
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from peft import PeftModel
        blip_processor = BlipProcessor.from_pretrained("blip_adapter")
        base = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base").to(DEVICE)
        blip_model = PeftModel.from_pretrained(base, "blip_adapter").to(DEVICE)
        blip_model.eval()
        print("  [BLIP+LoRA] loaded")
    except Exception as e: print(f"  BLIP: {e}"); traceback.print_exc()

val_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([.485, .456, .406], [.229, .224, .225])])

# ═══════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════
M5K   = "custom_5k    |  5,000 samples  |  BLEU-4: 4.34"
M100K = "custom_100k  |  100,000 samples  |  BLEU-4: 9.39  |  CIDEr: 0.85"
MBLIP = "blip_lora    |  LoRA fine-tuned  |  BLEU-4: 36.54  |  CIDEr: 1.35"
GREEDY = "greedy   — always picks top token"
BEAM   = "beam_5   — explores 5 paths  [recommended]"

def run(pil_img, model_choice, decode_method):
    if pil_img is None:
        return "$ awaiting input...", ""
    try:
        img = pil_img.convert("RGB")
        t0 = time.time()
        use_beam = "beam" in decode_method

        if "blip" in model_choice:
            if blip_model is None:
                return "$ error: blip_adapter/ not found", ""
            inputs = blip_processor(img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = blip_model.generate(**inputs, num_beams=5 if use_beam else 1,
                                          do_sample=False, max_length=50)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            dt = time.time() - t0
            return caption, f"model=blip_lora | decode={'beam_5' if use_beam else 'greedy'} | time={dt:.1f}s | device={DEVICE}"

        if "5k" in model_choice:
            model, mname = model_5k, "custom_5k"
        else:
            model, mname = model_100k, "custom_100k"
        if model is None:
            return f"$ error: {mname} checkpoint not found", ""

        t = val_tf(img)
        caption = model.generate_beam(t, k=5) if use_beam else model.generate_greedy(t)
        dt = time.time() - t0
        return caption, f"model={mname} | decode={'beam_5' if use_beam else 'greedy'} | time={dt:.1f}s | device={DEVICE}"

    except Exception as e:
        traceback.print_exc()
        return f"$ error: {e}", ""

MODELS = []
if model_5k: MODELS.append(M5K)
if model_100k: MODELS.append(M100K)
if blip_model: MODELS.append(MBLIP)
if not MODELS: MODELS = ["no models loaded"]
DEFAULT_M = M100K if model_100k else MODELS[0]

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #0C0C0C !important; color: #D4D4D4 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
footer { display: none !important; }
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }

#hdr { text-align: center; padding: 32px 0 22px; border-bottom: 1px solid #1A1A1A; margin-bottom: 28px; }
#hdr h1 { font-size: 1.1rem; font-weight: 400; letter-spacing: 0.22em;
           color: #4ADE80; margin: 0 0 6px; font-family: 'JetBrains Mono', monospace; }
#hdr .sub { font-size: 0.65rem; color: #777; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 14px; }
#hdr .badges { display: flex; justify-content: center; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.b { display: inline-block; background: #111; border: 1px solid #1E1E1E; color: #888;
     font-size: 0.6rem; padding: 3px 10px; border-radius: 3px; letter-spacing: 0.06em;
     font-family: 'JetBrains Mono', monospace; }
.b.g { border-color: #22C55E33; color: #4ADE80; }
.b.dim { border-color: #222; color: #666; }

.sl { font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase;
      color: #777; padding: 12px 0 8px; margin-bottom: 8px;
      font-family: 'JetBrains Mono', monospace; }
.sl .dot { color: #22C55E; }
.sl .cmt { color: #888; font-style: italic; letter-spacing: 0.08em; text-transform: none; }

.m-radio label, .d-radio label {
    background: #0F0F0F !important; border: 1px solid #1A1A1A !important;
    border-radius: 4px !important; color: #999 !important;
    font-size: 0.73rem !important; padding: 10px 14px !important;
    cursor: pointer; transition: all 0.2s;
    font-family: 'JetBrains Mono', monospace !important;
    display: block !important; width: 100% !important;
    margin: 3px 0 !important; line-height: 1.5 !important; }
.m-radio label:hover, .d-radio label:hover {
    border-color: #22C55E55 !important; color: #aaa !important; background: #0D1A0D !important; }
.m-radio input[type=radio]:checked + label, .d-radio input[type=radio]:checked + label,
.m-radio label:has(input:checked), .d-radio label:has(input:checked) {
    border-color: #22C55E !important; background: #0A1A0A !important; color: #4ADE80 !important; }

#gen-btn { background: #0A1A0A !important; border: 1px solid #22C55E !important;
           color: #4ADE80 !important; font-family: 'JetBrains Mono', monospace !important;
           font-size: 0.78rem !important; letter-spacing: 0.22em !important;
           border-radius: 4px !important; height: 48px !important;
           width: 100% !important; margin-top: 14px !important;
           cursor: pointer; transition: all 0.2s; }
#gen-btn:hover { background: #22C55E !important; color: #0C0C0C !important; }

#cap textarea { background: #080808 !important; border: 1px solid #1A1A1A !important;
    color: #22C55E !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.1rem !important; border-radius: 4px !important;
    min-height: 80px !important; padding: 14px !important;
    font-weight: 500 !important; text-shadow: 0 0 1px #22C55E44 !important; }
#cap textarea::placeholder { color: #333 !important; }
#cap .prose, #cap span, #cap div { color: #22C55E !important; }

#log textarea { background: #080808 !important; border: 1px solid #141414 !important;
    color: #888 !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important; border-radius: 3px !important;
    min-height: 30px !important; padding: 6px 10px !important; }

#info { background: #0D0D0D; border: 1px solid #181818; border-radius: 4px;
        padding: 16px; font-size: 0.7rem; color: #777; line-height: 2.1;
        margin-top: 14px; font-family: 'JetBrains Mono', monospace; }
#info b { color: #BBB; }
#info .g { color: #22C55E; font-weight: 700; }

#foot { text-align: center; padding: 24px 0 12px; border-top: 1px solid #161616;
        margin-top: 30px; font-family: 'JetBrains Mono', monospace; }
#foot .names { color: #888; font-size: 0.72rem; letter-spacing: 0.08em; margin-bottom: 4px; }
#foot .names span { color: #4ADE80; font-weight: 500; }
#foot .meta { color: #555; font-size: 0.6rem; letter-spacing: 0.1em; }
"""

with gr.Blocks(css=CSS, title="Image Captioning — FDL Demo") as demo:

    gr.HTML("""
    <div id="hdr">
      <h1>$ image_captioning --demo</h1>
      <div class="sub">foundations of deep learning · a.y. 2025/2026 · università di milano-bicocca</div>
      <div class="badges">
        <span class="b dim">5k &rarr; BLEU-4: 4.34</span>
        <span class="b">100k &rarr; BLEU-4: 9.39 · CIDEr: 0.85</span>
        <span class="b g">BLIP+LoRA &rarr; BLEU-4: 36.54 · CIDEr: 1.35</span>
        <span class="b dim">COCO 2014 · 29.3M params</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML('<div class="sl"><span class="dot">></span> INPUT_IMAGE  <span class="cmt">accepts jpg, png, heic, webp</span></div>')
            img_in = gr.Image(type="pil", label="", height=280, sources=["upload", "clipboard"])

            gr.HTML('<div class="sl" style="margin-top:18px"><span class="dot">></span> SELECT_MODEL</div>')
            model_radio = gr.Radio(choices=MODELS, value=DEFAULT_M,
                                   label="", show_label=False, elem_classes=["m-radio"])

            gr.HTML('<div class="sl" style="margin-top:14px"><span class="dot">></span> DECODING_METHOD</div>')
            decode_radio = gr.Radio(choices=[GREEDY, BEAM], value=BEAM,
                                    label="", show_label=False, elem_classes=["d-radio"])

            btn = gr.Button("> RUN INFERENCE", elem_id="gen-btn")

        with gr.Column(scale=1):
            gr.HTML('<div class="sl"><span class="dot">></span> GENERATED_CAPTION</div>')
            cap_out = gr.Textbox(value="$ awaiting input...",
                label="", show_label=False, lines=3, interactive=False, elem_id="cap")

            gr.HTML('<div class="sl" style="margin-top:12px"><span class="dot">></span> RUN_LOG  <span class="cmt">inference metadata</span></div>')
            log_out = gr.Textbox(value="", label="", show_label=False,
                lines=1, interactive=False, elem_id="log")

            gr.HTML("""
            <div id="info">
              <b>// ARCHITECTURE</b><br>
              <span class="g">></span> <b>custom_5k</b> — phase 1 only, frozen encoder, demonstrates data bottleneck<br>
              <span class="g">></span> <b>custom_100k</b> — phase 1 + 2, encoder blocks 6-7 unfrozen, two-phase transfer learning<br>
              <span class="g">></span> <b>blip_lora</b> — 249M total params, 1.77M trained (0.71%), pretrained on 129M image-text pairs<br>
              <span class="g">></span> <b>encoder</b> — EfficientNetV2-S (ImageNet-1K) to 49 patches x 256d + positional embedding<br>
              <span class="g">></span> <b>decoder</b> — 6-layer transformer, 8 heads, pre-layernorm, GELU, weight tying
            </div>
            """)

    gr.HTML("""
    <div id="foot">
      <div class="names">built by <span>Amin Entezari</span> & <span>Ali Sedghiye</span></div>
      <div class="meta">master's in data science · università degli studi di milano-bicocca · a.y. 2025/2026</div>
      <div class="meta" style="margin-top:3px">foundations of deep learning · EfficientNetV2 + Transformer · COCO 2014</div>
    </div>
    """)

    btn.click(fn=run, inputs=[img_in, model_radio, decode_radio],
              outputs=[cap_out, log_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
