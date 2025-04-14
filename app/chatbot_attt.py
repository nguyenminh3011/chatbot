import os
import pickle
import textwrap
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# === CẤU HÌNH ===
MODEL_NAME = "gemma2:9b"
DOCX_PATH = "Database.docx"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_FILE = "knowledge_hnsw.index"
CHUNKS_FILE = "chunks_hnsw.pkl"

# === ĐỌC FILE DOCX ===
def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])

# === CHUNK THÔNG MINH ===
def chunk_text_smart(text, max_length=1000):
    paragraphs = text.split('\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_length:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

# === TẢI EMBEDDING MODEL ===
print("[+] Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# === TẢI HOẶC TẠO FAISS INDEX + CHUNKS ===
if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    print("[+] Loading FAISS index and chunks from file...")
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
else:
    print(f"[+] Reading document from {DOCX_PATH}...")
    full_text = read_docx(DOCX_PATH)
    chunks = chunk_text_smart(full_text)
    print(f"[+] Total chunks: {len(chunks)}")

    print("[+] Generating embeddings...")
    embeddings = embed_model.encode(chunks)
    dimension = embeddings[0].shape[0]

    print("[+] Building FAISS HNSW index...")
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 là số neighbor
    index.hnsw.efSearch = 64
    index.add(np.array(embeddings))

    print(f"[+] Saving FAISS index to {INDEX_FILE} and chunks to {CHUNKS_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

# === TÌM ĐOẠN LIÊN QUAN ===
def search_relevant_chunks(question, top_k=3):
    question_vec = embed_model.encode([question])
    distances, indices = index.search(np.array(question_vec), top_k)
    return [chunks[i] for i in indices[0]]

# === KẾT NỐI OLLAMA ===
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Dùng khi chạy qua local
)

# === CHATBOT LOOP ===
print("=== Semantic Chatbot (HNSW + Smart Chunk) đã sẵn sàng! Gõ 'exit' để thoát. ===")
while True:
    user_input = input("\nYou: ")
    if user_input.lower().strip() == "exit":
        print("Tạm biệt!")
        break

    related_chunks = search_relevant_chunks(user_input)
    context = "\n".join(related_chunks)

    messages = [
        {"role": "system", "content": f"Dưới đây là thông tin liên quan:\n{context}"},
        {"role": "user", "content": user_input}
    ]

    print("Bot:", end=" ", flush=True)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        stream=True,
        messages=messages
    )

    bot_reply = ""
    for chunk in response:
        part = chunk.choices[0].delta.content or ""
        print(part, end="", flush=True)
        bot_reply += part

    print()
