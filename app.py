import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="centered"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .disclaimer {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 14px;
        margin-bottom: 16px;
    }
    .source-box {
        background: #f0f4ff;
        border-left: 3px solid #4a6fa5;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        color: #555;
        margin-top: 6px;
    }
    .triage-emergency { color: #d32f2f; font-weight: bold; }
    .triage-urgent    { color: #f57c00; font-weight: bold; }
    .triage-moderate  { color: #f9a825; font-weight: bold; }
    .triage-mild      { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Medical Knowledge Base
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    """Fever is defined as a body temperature above 38°C (100.4°F).
    Common causes include bacterial infections, viral infections like influenza,
    heat exhaustion, and inflammatory conditions. Mild fevers (38–39°C) can often
    be managed at home with rest and fluids. Seek medical attention if fever exceeds
    39.5°C, lasts more than 3 days, or is accompanied by severe headache, rash,
    or difficulty breathing.""",

    """Headaches are classified into primary and secondary types.
    Tension headaches are the most common, caused by stress or muscle tension.
    Migraines involve severe throbbing pain, often with nausea and light sensitivity.
    Cluster headaches cause intense pain around one eye. Red flag symptoms include
    sudden severe headache (thunderclap), headache with fever and stiff neck,
    or headache after head injury — these require immediate medical evaluation.""",

    """Chest pain can have cardiac and non-cardiac causes.
    Cardiac causes include angina and heart attack (myocardial infarction).
    Heart attack symptoms include crushing chest pain, pain radiating to left arm
    or jaw, sweating, and shortness of breath. Non-cardiac causes include acid reflux,
    costochondritis, and panic attacks. Any new chest pain should be evaluated urgently.
    Call emergency services immediately if pain is severe or with shortness of breath.""",

    """Shortness of breath (dyspnea) can result from respiratory or cardiac conditions.
    Common causes include asthma, COPD, pneumonia, heart failure, anemia, and anxiety.
    Sudden severe shortness of breath may indicate pulmonary embolism, pneumothorax,
    or acute heart failure. Seek emergency care if you cannot complete a sentence,
    lips turn blue, or breathing difficulty started suddenly without explanation.""",

    """Cough lasting less than 3 weeks is acute and often caused by viral upper
    respiratory infections. Chronic cough lasting over 8 weeks is commonly caused
    by postnasal drip, asthma, or GERD. A cough producing blood (hemoptysis)
    requires urgent medical evaluation to rule out tuberculosis, lung cancer,
    or pulmonary embolism.""",

    """Nausea and vomiting are symptoms of gastroenteritis, food poisoning, motion
    sickness, pregnancy, migraines, and medications. Gastroenteritis typically
    resolves within 1–3 days with rest and hydration. Seek medical care if vomiting
    lasts more than 24 hours, signs of dehydration are present, blood is in vomit,
    or vomiting is accompanied by severe abdominal pain.""",

    """Acute diarrhea lasting less than 2 weeks is usually caused by infections or
    food poisoning. Chronic diarrhea may indicate IBS, IBD, or celiac disease.
    Treatment includes oral rehydration. Seek care if diarrhea contains blood or mucus,
    is accompanied by high fever, or causes signs of dehydration.""",

    """Dizziness is described as lightheadedness, vertigo, or imbalance.
    Vertigo is often caused by BPPV, labyrinthitis, or Meniere's disease.
    Lightheadedness may result from dehydration, low blood pressure, or anemia.
    Dizziness with sudden hearing loss, facial weakness, or difficulty walking
    may indicate stroke and requires emergency evaluation.""",

    """Fatigue causes include insufficient sleep, anemia, thyroid disorders,
    diabetes, depression, and heart disease. Fatigue with unexplained weight loss,
    night sweats, or swollen lymph nodes requires thorough medical evaluation.""",

    """Lower back pain is mostly mechanical — caused by muscle strain, poor posture,
    or herniated discs. Red flags requiring urgent care include back pain with bladder
    or bowel dysfunction (cauda equina), pain after trauma, or unexplained weight loss.""",

    """Sore throat is most commonly caused by viral or bacterial infections.
    Strep throat requires antibiotics to prevent complications like rheumatic fever.
    Symptoms suggesting strep include sudden onset, fever, white patches on tonsils,
    and absence of cough. A throat swab confirms diagnosis.""",

    """Abdominal pain location helps identify the cause. Right lower quadrant pain
    may indicate appendicitis. Epigastric pain is often from gastritis or peptic ulcer.
    Sudden severe abdominal pain may indicate appendicitis, bowel perforation, or
    ectopic pregnancy and requires emergency evaluation.""",

    """Type 2 diabetes presents with increased thirst, frequent urination, unexplained
    weight loss, blurred vision, fatigue, and slow-healing wounds. Type 1 diabetes can
    present acutely with diabetic ketoacidosis causing fruity breath and vomiting.
    Diagnosis is confirmed by fasting blood glucose over 126 mg/dL or HbA1c over 6.5%.""",

    """High blood pressure is defined as systolic BP ≥ 130 mmHg or diastolic ≥ 80 mmHg.
    Long-term complications include stroke, heart attack, and kidney failure.
    Hypertensive emergency (BP > 180/120 with organ damage) requires immediate
    hospitalization.""",

    """Anaphylaxis is a severe allergic reaction causing throat swelling, difficulty
    breathing, drop in blood pressure, and loss of consciousness. It requires immediate
    epinephrine injection (EpiPen) and emergency care. Common triggers include peanuts,
    shellfish, medications, and insect stings.""",

    """Call emergency services (911/112) immediately for: chest pain or pressure,
    sudden difficulty breathing, signs of stroke (face drooping, arm weakness,
    speech difficulty — FAST acronym), severe allergic reaction, uncontrolled bleeding,
    loss of consciousness, seizures lasting more than 5 minutes, or severe head injury.""",
]

# ─────────────────────────────────────────────
# Load Models (cached so they load only once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_faiss_index(embedder):
    embeddings = embedder.encode(KNOWLEDGE_BASE, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index

@st.cache_resource(show_spinner=False)
def load_generator():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

# ─────────────────────────────────────────────
# RAG Functions
# ─────────────────────────────────────────────
def retrieve(question, index, embedder, top_k=3):
    vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(vec, top_k)
    return [
        {"chunk": KNOWLEDGE_BASE[i], "distance": float(d)}
        for d, i in zip(distances[0], indices[0])
    ]

def build_prompt(question, chunks):
    context = "\n\n".join([c["chunk"] for c in chunks])
    return f"""You are a helpful medical triage assistant.
Use the following medical information to answer the patient's question.
Be clear, concise, and always recommend consulting a doctor for diagnosis.

Medical Context:
{context}

Patient Question: {question}

Answer:"""

def generate_answer(prompt, tokenizer, model, device, max_new_tokens=256):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rag_pipeline(question, index, embedder, tokenizer, model, device, top_k=3):
    chunks  = retrieve(question, index, embedder, top_k)
    prompt  = build_prompt(question, chunks)
    answer  = generate_answer(prompt, tokenizer, model, device)
    sources = [c["chunk"][:150] + "..." for c in chunks]
    return answer, sources

# ─────────────────────────────────────────────
# UI — Header
# ─────────────────────────────────────────────
st.title("🏥 Medical RAG Chatbot")
st.markdown(
    '<div class="disclaimer">⚠️ <b>Disclaimer:</b> This chatbot is for '
    'educational purposes only. It is <b>NOT</b> a substitute for professional '
    'medical advice, diagnosis, or treatment. Always consult a qualified '
    'healthcare provider.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k        = st.slider("Chunks retrieved (top-k)", 1, 5, 3)
    show_sources = st.toggle("Show source chunks", value=True)
    st.divider()
    st.markdown("### 📚 Knowledge Base")
    st.write(f"{len(KNOWLEDGE_BASE)} medical topics loaded")
    st.markdown("""
    - Fever & infections
    - Chest pain & heart
    - Headache & neurology
    - Breathing problems
    - GI issues
    - Diabetes & BP
    - Emergencies
    """)
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# Load models with progress
# ─────────────────────────────────────────────
with st.spinner("⏳ Loading models (first run only — may take 1-2 mins)..."):
    embedder  = load_embedder()
    faiss_idx = load_faiss_index(embedder)
    tokenizer, gen_model, device = load_generator()

st.success(f"✅ Models ready! Running on: **{device}**", icon="🤖")

# ─────────────────────────────────────────────
# Chat history
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your medical triage assistant powered by RAG.\n\n"
                "I can help you:\n"
                "- 🔍 Check your symptoms\n"
                "- 🚦 Assess urgency level\n"
                "- 📋 Recommend next steps\n\n"
                "Please describe your symptoms and how long you've had them."
            ),
            "sources": [],
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander(f"📎 Sources used ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-box"><b>Chunk {i}:</b> {src}</div>',
                        unsafe_allow_html=True,
                    )

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
if question := st.chat_input("Describe your symptoms..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate RAG answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            answer, sources = rag_pipeline(
                question, faiss_idx, embedder, tokenizer, gen_model, device, top_k=top_k
            )

        st.markdown(answer)
        st.markdown("---")
        st.caption("⚠️ Always consult a doctor for medical decisions.")

        if show_sources and sources:
            with st.expander(f"📎 Sources used ({len(sources)} chunks)"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f'<div class="source-box"><b>Chunk {i}:</b> {src}</div>',
                        unsafe_allow_html=True,
                    )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
