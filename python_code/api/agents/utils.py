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


def get_embedding(model,text_input):
    embedding = model.encode([text_input], normalize_embeddings=True)
    embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
    return embedding
