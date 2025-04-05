from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent
                    )
from agents import AgentProtocol
import os

def main():
    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()
    
    agent_dict: dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
    }

    messages = []
    
    while True:
        # Display the chat history
        # os.system('cls' if os.name == 'nt' else 'clear')
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Get user input
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

    
        # Get Guard agent response
        response = guard_agent.get_response(messages)
        # print("Guard: ", response)
        if response["memory"]["guard_decision"] == "not allowed":
            messages.append(response)
            continue


        # Get classification agent
        response = classification_agent.get_response(messages)
        chosen_agent = response["memory"]["classification_decision"]
        print("Chosen Agent: ", chosen_agent)

        # get the chosen agent's response
        agent = agent_dict[chosen_agent]
        response = agent.get_response(messages)
        messages.append(response)


if __name__ == "__main__":
    main()

