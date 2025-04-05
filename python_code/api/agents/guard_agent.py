from openai import OpenAI
import os
from copy import deepcopy
from .utils import get_chat_response
from dotenv import load_dotenv
import json
load_dotenv()

client = OpenAI(
    api_key=os.getenv("TOKEN"),
    base_url=os.getenv("BASE_URL"),
)
model_name = os.getenv("MODEL_NAME")