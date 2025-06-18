# Q-A_generator
A Streamlit app that lets users upload a PDF, generate topic-based questions and answers using Groq and Ollama, and download them as separate PDFs.

# 📘 Question-Answer Generator using Groq, Ollama & Streamlit

This is a **Streamlit-based web app** that allows users to:

- Upload a **PDF document** (e.g., lecture notes or study material)
- Enter a **topic**, select a **difficulty level**, and choose a **question type**
- Automatically generate **5 high-quality questions and answers** using:
  - 📎 Ollama embeddings (`mxbai-embed-large`)
  - 🤖 Groq LLMs (e.g., `gemma2-9b-it`)
- Download the generated **Questions** and **Answers** as separate **PDF files**

---

## 🚀 Features

- 📤 PDF upload by user
- 📌 Topic, difficulty, and question type selection
- 🧠 Uses Groq’s LLM for content generation
- 📎 Vector storage with Chroma and Ollama
- 📄 Downloadable PDFs for Questions and Answers
- ⏱️ Fast response time display

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Ollama embeddings**
- **Groq API**
- **Chroma Vector Store**
- **PyPDFLoader**
- **Fpdf2** for PDF generation

---

## 📦 Installation


git clone https://github.com/yourusername/qa-generator-groq.git
cd qa-generator-groq
python -m venv venv
venv\Scripts\activate  # For Windows
pip install -r requirements.txt
