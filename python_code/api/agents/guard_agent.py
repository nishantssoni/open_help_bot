from openai import OpenAI
import os
from copy import deepcopy
from .utils import get_chat_response
from dotenv import load_dotenv
import json
load_dotenv()



class GuardAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("TOKEN"),
            base_url=os.getenv("BASE_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")
    
    def get_response(self, messages):
        messages = deepcopy(messages)

        system_prompt = """
            You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
            Your ask is to determine whether the user is asking something relevant to the coffee shop or not.
            
            The user is allowed to:
            1. Ask questions about the coffee shop, like location, working hours, menu items and coffee show releated questions.
            2. Ask questions about menu items, they can ask for ingredients in an item and more details about the item.
            3. Make and order.
            4. Ask about recommendations of what to buy.

            The user is not allowed to:
            1. Ask anything that is not related to the coffee shop.
            2. ask questions about the staff or how to make a certain menu items.

            Your output should be in a structured json format like so. each key is a string and each value is a string. exactly like the one bellow.  You are not allowed to write anythinkg other than the json object and make sure to follow the format exactly
            {
            "chain of thought": "go over each of the points above and see if the message lies under this point or not. Then you write some thoughts about what point is this input relevant to."
            "decision": "'allowed' or 'not allowed'. Pick one of those and only write the word."
            "message": "give response if it's allowed, otherwise write 'Sorry, I can't help with that. Can I help you with your order?'"
            }
        """
        input_messages = [{"role": "system", "content": system_prompt}]  + messages[-3:]

        chatbot_response = get_chat_response(self.client, self.model_name, input_messages,max_tokens=1000)
        output = self.postprocess(chatbot_response)

        return output
    
    def postprocess(self, response):
        output = json.loads(response)
        
        dict_output = {
            "role": "assistant",
            "content": output["message"],
            "memory":{
                "agent":"gurad_agent",
                "guard_decision":output["decision"],
            }
        }

        return dict_output