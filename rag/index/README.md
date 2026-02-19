# RAG Index Directory

Vector database index files are stored here.

## Contents After Building

```
index/
├── documents.json     # Document metadata
├── embeddings.npy     # Document embeddings
├── faiss.index        # FAISS index file
└── chromadb/          # ChromaDB files (if using Chroma)
```

## Building the Index

```bash
python -m ai_voice_detection.rag.build_index
```

Or in Python:

```python
from ai_voice_detection.rag import build_vector_index

builder = build_vector_index(output_path="rag/index")
```

## Adding Custom Documents

```python
from ai_voice_detection.rag import KnowledgeBaseBuilder

builder = KnowledgeBaseBuilder()
builder.add_documents([
    {
        "id": "custom_doc_1",
        "title": "Custom Research Paper",
        "category": "research",
        "text": "Your custom document content..."
    }
])
builder.build_index()
builder.save()
```
