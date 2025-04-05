from openai import OpenAI
import os
from copy import deepcopy
from .utils import get_chat_response, get_embedding
from dotenv import load_dotenv
import json
load_dotenv()



class ClassificationAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("TOKEN"),
            base_url=os.getenv("BASE_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")
    
    def get_response(self, messages):
        messages = deepcopy(messages)