import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fpdf import FPDF


st.title("📚 Question-Answers Generator with Groq")

uploaded_file = st.file_uploader("📄 Upload your source PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_file_path = os.path.join("temp_uploaded.pdf")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        st.session_state.loader = PyPDFLoader(temp_file_path)
        st.session_state.doc = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.doc)
        st.session_state.vectors = chroma.Chroma.from_documents(
            st.session_state.final_doc, st.session_state.embeddings
        )

    llm = ChatGroq(
        groq_api_key="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",
        model_name="gemma2-9b-it"
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an intelligent tutor and content generator.

Given the following:
- A document: {context}
- A topic of focus: "{topic}"
- A difficulty level: "{difficulty_level}" (choose from Easy, Medium, or High)
- A question type: "{question_type}" (choose from "MCQs", "Short Answer Questions", or "Long Answer Questions")

Your task is to:
1. Understand the document and extract key information related to the topic.
2. Generate 5 high-quality questions based on the topic and document content.
3. Ensure the questions match the requested difficulty level and question type.
4. Provide a clear and accurate answer for each question.

### Output Format:

**Question 1:** [Question here]  
**Answer:** [Answer here]

**Question 2:** [Question here]  
**Answer:** [Answer here]

...and so on up to 5 questions.

Make the questions educational, relevant to the topic, and aligned with the specified difficulty and type.
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    topic = st.text_input("Enter the topic (e.g., Neural Networks)")
    difficulty = st.selectbox("Select difficulty level", ["Easy", "Medium", "High"])
    question_type = st.selectbox("Select question type", ["MCQs", "Short Answer Questions", "Long Answer Questions"])

    if topic and difficulty and question_type:
        if st.button("Generate Questions"):
            start = time.process_time()
            response = retrieval_chain.invoke({
                "input": topic,
                "topic": topic,
                "difficulty_level": difficulty,
                "question_type": question_type
            })
            output_text = response['answer']
            st.write("✅ **Generated Output:**")
            st.write(output_text)

            # Separate questions and answers
            questions = []
            answers = []
            lines = output_text.split('\n')
            for line in lines:
                if line.strip().startswith("**Question"):
                    questions.append(line.strip())
                elif line.strip().startswith("**Answer:**"):
                    answers.append(line.strip())

            # Generate PDF files
            def generate_pdf(text_list, title):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, title + "\n\n")
                for item in text_list:
                    pdf.multi_cell(0, 10, item + "\n")
                return pdf.output(dest="S").encode("latin-1")

            question_pdf = generate_pdf(questions, "Generated Questions")
            answer_pdf = generate_pdf(answers, "Generated Answers")

            st.download_button("📥 Download Questions PDF", question_pdf, file_name="questions.pdf")
            st.download_button("📥 Download Answers PDF", answer_pdf, file_name="answers.pdf")

            print("Response time:", round(time.process_time() - start), "seconds.")
