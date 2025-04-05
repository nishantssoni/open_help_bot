from openai import OpenAI
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import faiss
import numpy as np
from dotenv import load_dotenv
load_dotenv()


def get_chat_response(client, model_name, messages, temprature=0.0, top_p=0.8, max_tokens=100):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"], "content": message["content"]})

    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=temprature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


def get_embedding(client, model,text_input):
    embedding = model.encode([text_input], normalize_embeddings=True)
    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
    return embedding



def retrieve_and_respond(client, llm_model, query, embedding_model, index_file_name, data, top_k=5):
    """
    Retrieve relevant documents based on query and generate a response.
    
    Parameters:
    query (str): User query.
    model (object): Model to generate embeddings.
    index_file (faiss.IndexFlatL2): FAISS index filename containing stored embeddings.
    texts (list of str): Original text data corresponding to stored embeddings.
    top_k (int): Number of top results to retrieve.
    """
    # Generate query embedding
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)



    # Load FAISS index
    index = faiss.read_index(index_file_name)
    
    # Retrieve top-k similar documents
    D, I = index.search(query_embedding, top_k)
    retrieved_docs = [data[i] for i in I[0]]
    context = "\n".join(retrieved_docs)
    
    # Generate response using OpenAI API
    messages=[{"role": "system", "content": "You are an AI assistant with domain expertise."},
                  {"role": "user", "content": f"Using this information: {context},answer this question: {query}."}]
    
    print(messages)

    response = get_chat_response(client, llm_model, messages)
    
    return response

# retrieve_and_respond(client,model_name,"how about i phone 16 camera", model, "bge_vector_store.index", data, top_k=1)