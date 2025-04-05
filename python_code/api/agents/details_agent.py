from openai import OpenAI
import os
import pickle
from copy import deepcopy
import faiss
from .utils import get_chat_response, get_embedding
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
load_dotenv()



class DetailsAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("TOKEN"),
            base_url=os.getenv("BASE_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")
        self.index_file_name = "faiss_product.index"
        print("current directory is : ", os.getcwd())

        # Download the model to a local directory
        self.local_model_path = "bge-small-en"
        # download the model
        # snapshot_download(repo_id="BAAI/bge-small-en", local_dir=self.local_model_path)

        # Now load the locally saved model
        self.embedding_model = SentenceTransformer(self.local_model_path)

        # load the data
        # Load from pickle file
        with open("data.pkl", "rb") as f:
            self.data = pickle.load(f)

    
    def get_response(self, messages):
        messages = deepcopy(messages)
        user_message = messages[-1]['content']

        # Generate query embedding\
        query_embedding = get_embedding(self.embedding_model, user_message)

        # Load FAISS index
        index = faiss.read_index(self.index_file_name)

        # Retrieve top-k similar documents
        D, I = index.search(query_embedding, 1)  # k=1
        retrieved_docs = [self.data[i] for i in I[0]]
        context = "\n".join(retrieved_docs)
        print("context is : ", context)

        prompt = f"""
            Using the contexts below, answer the query.

            Contexts:
            {context}

            Query: {user_message}
            """
        system_prompt = """ You are a customer support agent for a coffee shop called Merry's way. You should answer every question as if you are waiter and provide the neccessary information to the user regarding their orders """
        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chat_response(self.client,self.model_name,input_messages)
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self,output):
        output = {
            "role": "assistant",
            "content": output,
            "memory": {"agent":"details_agent"
                      }
        }
        return output