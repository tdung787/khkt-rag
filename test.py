results = retriever.search(
    query="Gen A dài 0,5 µm nucleotit adenin đột biến alen a2",
    subject="Sinh học",
    top_k=5
)

# Xem có câu cau_116 không?
for r in results:
    print(f"{r['question_id']}: {r['score']:.4f}")