import streamlit as st
import pickle
import random
import math

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BashBrain",
    page_icon="🧠",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #e6edf3;
}

.stApp {
     background: radial-gradient(
        circle at 20% 20%,
        #065f46,
        transparent 50%
    ),
    radial-gradient(
        circle at 80% 80%,
        #064e3b,
        transparent 50%
    ),
    linear-gradient(
        135deg,
        #022c22,
        #064e3b,
        #065f46
    );
}

h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.4rem !important;
    color: rgba(255, 255, 255, 0.92) !important;
    letter-spacing: -1px;
}

.subtitle {
    color: rgba(255, 255, 255, 0.92);
    font-size: 1rem;
    margin-top: -10px;
    margin-bottom: 30px;
}

.stTextInput > div > div > input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15) !important;
}

.stButton > button {
    /* Glass base */
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.25),
        rgba(255, 255, 255, 0.08)
    ) !important;

    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);

    /* Border + depth */
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 14px !important;

    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.25),
        inset 0 1px 0 rgba(255, 255, 255, 0.4);

    /* Text */
    color: rgba(255, 255, 255, 0.95) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;

    /* Layout */
    padding: 10px 24px !important;
    width: 100%;
    transition: all 0.25s ease;
}

/* Hover state */
.stButton > button:hover {
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.35),
        rgba(255, 255, 255, 0.15)
    ) !important;

    box-shadow:
        0 12px 40px rgba(0, 0, 0, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.6);

    transform: translateY(-2px);
}

/* Active (pressed) state */
.stButton > button:active {
    transform: translateY(0px);
    box-shadow:
        0 4px 20px rgba(0, 0, 0, 0.3),
        inset 0 2px 6px rgba(0, 0, 0, 0.2);
}

.result-box {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: #e6edf3;
    transition: border-color 0.2s;
}

.result-box:hover {
    border-left-color: #3fb950;
}

.prompt-highlight {
    color: #58a6ff;
    font-weight: 700;
}

.completion-text {
    color: #3fb950;
}

.section-label {
    color: #8b949e;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    margin-top: 24px;
}

.info-box {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 40px;
    color: #8b949e;
    font-size: 0.85rem;
    line-height: 1.6;
}

.info-box a {
    color: #58a6ff;
    text-decoration: none;
}

.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 30px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        state_dict = pickle.load(f)
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return state_dict, vocab

try:
    state_dict, vocab = load_model()
    uchars = vocab['uchars']
    BOS = vocab['BOS']
    vocab_size = vocab['vocab_size']
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Model functions (copied from microgpt.py) ──────────────────────────────────
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-Value(other) if not isinstance(other, Value) else -other)
    def __rsub__(self, other): return Value(other) + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return Value(other) * self**-1

n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

def linear(x, w):
    return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - Value(max_val)).exp() for val in logits]
    total = sum(exps, Value(0))
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum((xi * xi for xi in x), Value(0)) * Value(1.0 / len(x))
    scale = (ms + Value(1e-5)) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum((q_h[j] * k_h[t][j] for j in range(head_dim)), Value(0)) * Value(head_dim**-0.5) for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum((attn_weights[t] * v_h[t][j] for t in range(len(v_h))), Value(0)) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

def generate(prompt="", temperature=0.5, num_samples=5):
    results = []
    for _ in range(num_samples):
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]

        # Feed in BOS first
        token_id = BOS
        logits = gpt(token_id, 0, keys, vals)

        # Feed in prompt characters
        prompt_tokens = [uchars.index(ch) for ch in prompt if ch in uchars]
        for pos_id, tok in enumerate(prompt_tokens):
            token_id = tok
            if pos_id < len(prompt_tokens) - 1:
                gpt(token_id, pos_id + 1, keys, vals)

        # Generate continuation
        generated = list(prompt)
        start_pos = len(prompt_tokens)
        for pos_id in range(start_pos, block_size):
            logits = gpt(token_id, pos_id, keys, vals)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            generated.append(uchars[token_id])

        results.append(''.join(generated))
    return results

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("# BashBrain")
st.markdown('<p class="subtitle">A tiny GPT trained from scratch that autocompletes bash commands</p>', unsafe_allow_html=True)

if not model_loaded:
    st.error("model.pkl or vocab.pkl not found. Run `python train.py` first.")
    st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    prompt = st.text_input(
        "Start of a bash command",
        placeholder="e.g.  find . -name   or   grep -r   or   tar -",
        label_visibility="collapsed"
    )
with col2:
    temperature = st.slider("Creativity", 0.1, 1.0, 0.5, 0.1, label_visibility="collapsed")

run = st.button("Complete command", use_container_width=True)

if run or prompt:
    if not prompt.strip():
        st.warning("Type the start of a bash command above.")
    else:
        with st.spinner("Thinking..."):
            suggestions = generate(prompt.strip(), temperature=temperature, num_samples=5)

        st.markdown('<p class="section-label">Suggestions</p>', unsafe_allow_html=True)
        for s in suggestions:
            completion = s[len(prompt):]
            st.markdown(
                f'<div class="result-box">'
                f'<span class="prompt-highlight">{prompt}</span>'
                f'<span class="completion-text">{completion}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong style="color:#e6edf3">How it works</strong><br>
    BashBrain uses a tiny GPT implemented in pure Python — no PyTorch, no NumPy.
    The model was trained on ~11,000 real bash commands from the
    <a href="https://github.com/TellinaTool/nl2bash">NL2Bash dataset</a>,
    learning bash syntax character by character from scratch.<br>
    The Creativity slider controls temperature: low = predictable output, high = more varied and experimental. <br><br>
    Built on <a href="https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95">microgpt.py</a>
    by Andrej Karpathy.
</div>
""", unsafe_allow_html=True)