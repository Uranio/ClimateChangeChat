import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Configure the page
st.set_page_config(page_title="Climate Change Chatbot", page_icon="💬")

# Sidebar for configuration (API Key)
with st.sidebar:
    st.title("Climate Change Chatbot")
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        #st.success("API Key found!", icon="✅")
    else:
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        if not api_key:
            st.warning("Please enter your API Key to continue.", icon="⚠️")
    st.markdown("This app answers your questions about climate change. It can make mistakes. Consider verifying all the informations.")

# Stop the app if the API Key is not provided
if not api_key:
    st.stop()

# --- Loading and preparing the FAQ ---
file_path = 'climate_change_faqs.csv'
df = pd.read_csv(file_path)

# Extract questions and answers, ensuring the structure is correct
faq_pairs = []
current_question = None

for _, row in df.iterrows():
    if row["text_type"] == "q":
        current_question = row["faq"]
    elif row["text_type"] == "a" and current_question:
        faq_pairs.append({
            "question": current_question,
            "answer": row["faq"],
            "source": row["source"]
        })
        current_question = None

# Convert to structured DataFrame
faq_df = pd.DataFrame(faq_pairs)

# Extract the questions and answers
faq_questions = faq_df['question'].tolist()
faq_answers = faq_df['answer'].tolist()
faq_sources = faq_df['source'].tolist()

# Create embeddings for the questions and build the FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(faq_questions, convert_to_numpy=True)
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# --- Initialize the OpenAI client ---
client = OpenAI(api_key=api_key)

# --- Set up the chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are an assistant that answers exclusively using the FAQ data provided. Do not add external information."}
    ]

# --- Function to retrieve FAQ context based on the query ---
def retrieve_faq(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved = []
    for idx in indices[0]:
        retrieved.append({
            "question": faq_questions[idx],
            "answer": faq_answers[idx],
            "source": faq_sources[idx]
        })
    return retrieved

# --- Function to generate the answer (using OpenAI) ---
def generate_answer(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the desired model if necessary
        messages=messages,
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content

# --- Display the chat history ---
for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# --- User input ---
if user_input := st.chat_input("Write your message:"):
    # Retrieve the most relevant FAQs for the current message
    retrieved = retrieve_faq(user_input, top_k=3)
    faq_context = "\n\n".join(
        f"FAQ {i+1}:\nQ: {faq['question']}\nA: {faq['answer']}\n(Source: {faq['source']})" 
        for i, faq in enumerate(retrieved)
    )
    
    # Build the user message including the FAQ context
    user_message_content = (
        "Use exclusively the following FAQ data to answer:\n\n"
        f"{faq_context}\n\n"
        f"Question: {user_input}\n"
        "Answer:"
    )
    
    # Add the user's message to the chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_message_content})
    with st.chat_message("user"):
        with st.expander("Sources"):
            st.write(user_message_content)
    
    # Generate the assistant's response based on the entire chat history (including the FAQ context)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_response = generate_answer(st.session_state.messages)
            st.write(assistant_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
