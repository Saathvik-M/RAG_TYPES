RAG_Playground/
│
├── backend/
│   │
│   ├── app.py
│   ├── config.py
│   ├── .env
│   ├── requirements.txt
│   │
│   ├── documents/
│   │
│   ├── vectorstore/
│   │
│   ├── routers/
│   │      chat.py
│   │      upload.py
│   │
│   ├── rag/
│   │      naive.py
│   │      advanced.py
│   │      hybrid.py
│   │      agentic.py
│   │
│   ├── services/
│   │      pdf_loader.py
│   │      chunker.py
│   │      embeddings.py
│   │      vector_store.py
│   │      retriever.py
│   │      llm.py
│   │
│   └── models/
│          request_models.py
│
├── frontend/
│      streamlit_app.py
│
└── README.md