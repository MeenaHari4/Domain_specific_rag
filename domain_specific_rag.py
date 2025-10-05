import streamlit as st
import numpy as np
import pickle
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load precomputed data ----
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

text_embeddings = np.load("text_embeddings.npy")

st.title("📚 Domain-Specific RAG with Mistral")

client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Retrieving answer..."):
        # 1️⃣ Embed query
        q_embed = client.embeddings.create(model="mistral-embed", inputs=[query])
        query_embedding = np.array(q_embed.data[0].embedding)

        # 2️⃣ Retrieve relevant chunks
        similarities = cosine_similarity([query_embedding], text_embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        context = "\n\n".join([chunks[i] for i in top_indices])

        # 3️⃣ Generate answer using Mistral chat
        prompt = f"""
        You are a domain expert assistant. Use only the context below to answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        st.markdown("### 🧠 Answer:")
        st.write(response.choices[0].message.content)
