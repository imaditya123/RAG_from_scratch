# 📖 RAG from Scratch  

A minimal, end-to-end implementation of a **Retrieval-Augmented Generation (RAG)** pipeline built from scratch.  
This project demonstrates how to combine **information retrieval** with **large language models (LLMs)** to build smarter, context-aware applications.  

---

## 🚀 Features  
- Build a **document store** to index and manage knowledge sources  
- Implement **text chunking, embeddings, and vector search** for efficient retrieval  
- Integrate retrieval results with an **LLM for augmented responses**  
- Modular and easy-to-extend codebase (no heavy frameworks required)  
- Step-by-step implementation for learning & experimentation  

---

## 📂 Project Structure  
```
.
├── main.py             # Entry point for running the RAG pipeline
├── rag_document.py     # Document representation & utilities
├── retriever.py        # Embedding + vector search
├── generator.py        # LLM integration
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## 🧑‍💻 Getting Started  

### 1. Clone the repo  
```bash
git clone https://github.com/imaditya123/RAG_from_scratch.git
cd RAG_from_scratch
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline  
```bash
python main.py
```

---

## 🎯 Goals  
- Learn how RAG works **under the hood**  
- Provide a **minimal reference implementation** for students, researchers, and hobbyists  
- Build a foundation that can be extended into production-ready RAG systems  

---

## 📝 Future Improvements  
- Add support for multiple vector databases (FAISS, Pinecone, etc.)  
- Support for multi-modal documents (PDFs, images)  
- Evaluation metrics for retrieval + generation quality  

---

## 🤝 Contributing  
Contributions are welcome! Feel free to open issues or submit pull requests with improvements and new features.  

---

## 📜 License  
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.  
