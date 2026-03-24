import os
import json
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../../chunks_contextual.json")
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "../../database/qdrant_contextual")
COLLECTION_NAME = "digital_modulation_contextual"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_ACRONYM_LIST = (
    "FSK=Frequency Shift Keying, PSK=Phase Shift Keying, BPSK=Binary Phase Shift Keying, "
    "QPSK=Quadrature Phase Shift Keying, DQPSK=Differential Quadrature Phase Shift Keying, "
    "OQPSK=Offset Quadrature Phase Shift Keying, QAM=Quadrature Amplitude Modulation, "
    "MSK=Minimum Shift Keying, GMSK=Gaussian Minimum Shift Keying, GFSK=Gaussian Frequency Shift Keying, "
    "BFSK=Binary Frequency Shift Keying, FFSK=Fast Frequency Shift Keying, "
    "MPSK=Minimum Phase Shift Keying, AM=Amplitude Modulation, FM=Frequency Modulation, "
    "PCM=Pulse Code Modulation, ADPCM=Adaptive Digital Pulse Code Modulation, "
    "OFDM=Orthogonal Frequency Division Multiplexing, COFDM=Coded Orthogonal Frequency Division Multiplexing, "
    "TDMA=Time Division Multiple Access, CDMA=Code Division Multiple Access, "
    "FDMA=Frequency Division Multiple Access, "
    "BER=Bit Error Rate, FER=Frame Error Rate, EVM=Error Vector Magnitude, "
    "ACP=Adjacent Channel Power, ISI=Intersymbol Interference, BW=Bandwidth, PSD=Power Spectral Density, "
    "GSM=Global System for Mobile Communication, AMPS=Advanced Mobile Phone System, "
    "NADC=North American Digital Cellular, DECT=Digital Enhanced Cordless Telephone, "
    "TETRA=Trans European Trunked Radio, "
    "DVB-C=Digital Video Broadcast Cable, DVB-S=Digital Video Broadcast Satellite, "
    "DVB-T=Digital Video Broadcast Terrestrial, DAB=Digital Audio Broadcast, "
    "I/Q=In-phase Quadrature, IF=Intermediate Frequency, RF=Radio Frequency, "
    "DSP=Digital Signal Processing, FFT=Fast Fourier Transform, "
    "FDD=Frequency Division Duplex, TDD=Time Division Duplex"
)

SYSTEM_PROMPT = """Your task is to answer the user's question using ONLY the provided context. Always respond in the same language as the user's question.

## WHAT YOU MUST COVER
Include ALL key information from the matched section:
- Definitions and concept relationships
- Technical details (formulas, numeric values, parameters)
- Applications and examples

## WHAT YOU CAN AND CANNOT DO
You MAY:
- Paraphrase for clarity
- Reorganize information logically
- Apply formulas step-by-step if the user asks a calculation question

You MUST NOT:
- Omit any information from the section
- Add external knowledge or speculate beyond the context

If the context lacks enough information, respond exactly with:
"Insufficient information in the provided context."

## FORMATTING RULES

**Bold:** Use ONLY for a key term or technical name, ONLY on its first appearance, ONLY if 4 words or fewer.
- ✅ **symbol clock** is the frequency and timing used to transmit each symbol.
- ❌ **Symbol clock is the frequency and timing used to transmit each symbol.**

**Bullet points:** Use when listing 3+ parallel items, trade-offs, or features — not for single ideas or flowing explanation.

**Numbered lists:** Use for ordered steps or processes where sequence matters.

**Headings:** Use a short plain heading only when the answer has 3+ clearly distinct sections. Skip headings for short or single-topic answers.

**Paragraphs:** 2–4 sentences max. One blank line between paragraphs. Never break a paragraph mid-sentence just to add whitespace.

**What NOT to do:**
- Do not bold phrases longer than 4 words
- Do not put each sentence on its own line
- Do not add a heading for every paragraph
- Do not mix bullet points and prose randomly

## RESPONSE STRUCTURE

1. Start with 1–2 sentences giving the direct answer.
2. Then explain details — use prose for concepts, numbered lists for processes, bullets for enumerations.
3. End with the reference block.

## EXAMPLES

---
Question: What is digital modulation?

Answer:
**Digital modulation** is the process of encoding digital data onto a carrier wave so it can be transmitted efficiently over a communication channel.

Modern systems use it to optimize three main trade-offs:
- **Bandwidth efficiency** — fitting more data into limited spectrum
- **Power efficiency** — transmitting reliably at the lowest practical power
- **Cost efficiency** — minimizing hardware and deployment costs

Different systems prioritize differently. Terrestrial microwave radios favor bandwidth efficiency and low BER, while mobile handsets prioritize power (battery life) and cost, sometimes sacrificing bandwidth efficiency.

Reference:
- Section 1 – Why Digital Modulation? (Page 5)

---
Question: How does a digital communications receiver work?

Answer:
A digital receiver reconstructs the original bit stream from an incoming RF signal, which has been degraded by noise, interference, and multipath fading.

The demodulation process follows these steps:
1. Recover the carrier frequency (carrier lock)
2. Recover the symbol clock (symbol lock)
3. Split the signal into I and Q components
4. Slice I/Q values to determine each symbol
5. Decode and de-interleave
6. Reconstruct the original bit stream
7. Convert back to analog if needed

The key challenge versus the transmitter is clock and carrier recovery. Even if the frequency is correct, a misaligned symbol clock phase — sampling at a transition edge instead of the symbol center — will cause demodulation to fail.

Reference:
- Section 7.2 – A digital communications receiver (Page 36)

---

## REFERENCE FORMAT
At the end of every answer, include:

Reference:
- Section X.X – [Title] (Page Y)

List each section on a separate line if multiple sections were used.
"""


_instance: "RAGTool | None" = None


def get_rag_tool() -> "RAGTool":
    """Return the global RAGTool singleton (one QdrantClient for the whole process)."""
    global _instance
    if _instance is None:
        _instance = RAGTool()
    return _instance


class RAGTool:
    """Semantic RAG tool using Qdrant only (no Elasticsearch)."""

    def __init__(self):
        self.openai = OpenAI()

        # Load chunks
        chunks_path = os.path.normpath(CHUNKS_PATH)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}

        # Load embedding model
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to Qdrant
        db_path = os.path.normpath(QDRANT_PATH)
        self.qdrant = QdrantClient(path=db_path)
        self._init_qdrant()

    def _init_qdrant(self):
        """Initialize Qdrant collection if not exists or count mismatch."""
        if self.qdrant.collection_exists(COLLECTION_NAME):
            info = self.qdrant.get_collection(COLLECTION_NAME)
            if info.points_count == len(self.chunks):
                return  # Already up to date

        print("Building Qdrant collection from chunks_contextual.json...")
        embedding_dim = self.encoder.get_sentence_embedding_dimension()

        if self.qdrant.collection_exists(COLLECTION_NAME):
            self.qdrant.delete_collection(COLLECTION_NAME)

        self.qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

        texts = [
            f"{c['text']}\n\n{c.get('contextualized_content', '')}"
            for c in self.chunks
        ]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        points = [
            PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "contextualized_content": chunk.get("contextualized_content", ""),
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "section_title": chunk["section_title"],
                    "chapter": chunk["chapter"],
                    "chapter_title": chunk["chapter_title"],
                    "primary_concepts": chunk.get("primary_concepts", []),
                    "technical_terms": chunk.get("technical_terms", []),
                    "content_type": chunk.get("content_type", ""),
                    "difficulty": chunk.get("difficulty", ""),
                    "has_formula": chunk.get("has_formula", False),
                    "has_example": chunk.get("has_example", False),
                },
            )
            for idx, (chunk, emb) in enumerate(zip(self.chunks, embeddings))
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=COLLECTION_NAME, points=points[i : i + batch_size]
            )
        print(f"Indexed {len(points)} vectors.")

    def expand_query(self, query: str) -> str:
        """Expand acronyms and translate Vietnamese query to English for better retrieval."""
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rewrite a search query for a digital modulation textbook (written in English).\n\n"
                            f"Glossary: {_ACRONYM_LIST}\n\n"
                            "Rules:\n"
                            "1. If the query is in Vietnamese, translate it to English first.\n"
                            "2. If the query contains acronyms from the glossary, keep the acronym AND add its full form: fsk → FSK Frequency Shift Keying\n"
                            "3. Handle any casing: fsk, Fsk, FSK → FSK Frequency Shift Keying\n"
                            "4. Do NOT expand common English words: 'if', 'am', 'or' are NOT acronyms\n"
                            "5. Only expand acronyms from the glossary, do NOT invent expansions\n"
                            "6. Return ONLY the rewritten query, nothing else"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=200,
            )
            expanded = response.choices[0].message.content.strip()
            if expanded:
                return expanded
        except Exception:
            pass
        return query

    def retrieve(self, query: str, k: int = 7) -> List[Dict]:
        """Semantic search using Qdrant."""
        expanded = self.expand_query(query)
        vector = self.encoder.encode(expanded).tolist()
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=k,
        )
        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "score": r.score,
                "text": r.payload["text"],
                "contextualized_content": r.payload.get("contextualized_content", ""),
                "page": r.payload["page"],
                "section": r.payload["section"],
                "section_title": r.payload["section_title"],
                "chapter": r.payload["chapter"],
                "chapter_title": r.payload["chapter_title"],
                "content_type": r.payload.get("content_type", ""),
            }
            for r in results
        ]

    def retrieve_by_topic(self, topic: str, k: int = 10) -> List[Dict]:
        """Retrieve chunks relevant to a topic — used by quiz generator."""
        return self.retrieve(topic, k=k)

    def build_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks as a context string for LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Page {chunk['page']}, Section {chunk['section']}: {chunk['section_title']}\n"
                f"Chapter {chunk['chapter']}: {chunk['chapter_title']}\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, context: str, stream: bool = False):
        """Generate answer from context. Returns str or generator if stream=True."""
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        if stream:
            def _stream():
                response = self.openai.chat.completions.create(
                    model="gpt-5.2-2025-12-11",
                    messages=messages,
                    stream=True,
                )
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
            return _stream()

        response = self.openai.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=messages,
        )
        return response.choices[0].message.content

    def generate_casual(self, query: str, stream: bool = False):
        """Generate friendly response for casual/non-technical messages."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly assistant specialized in digital modulation & communications.\n"
                    "The user sent a casual message. Respond naturally and briefly.\n"
                    "If appropriate, let them know you can help with questions about digital modulation.\n"
                    "Respond in the same language the user uses."
                ),
            },
            {"role": "user", "content": query},
        ]

        if stream:
            def _stream():
                try:
                    response = self.openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=300,
                        stream=True,
                    )
                    for chunk in response:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            yield delta
                except Exception:
                    yield "Xin chào! Tôi có thể giúp bạn với các câu hỏi về điều chế số."
            return _stream()

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception:
            return "Xin chào! Tôi có thể giúp bạn với các câu hỏi về điều chế số."

    def answer(self, query: str, stream: bool = False):
        """Full RAG pipeline: retrieve → build context → generate."""
        chunks = self.retrieve(query)
        context = self.build_context(chunks)
        return self.generate(query, context, stream=stream)
