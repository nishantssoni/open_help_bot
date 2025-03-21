from openai import OpenAI
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import faiss
import numpy as np


load_dotenv()


# Initialize OpenAI API
client = OpenAI(
    api_key=os.getenv("TOKEN"),
    base_url=os.getenv("BASE_URL"),
)
model_name = os.getenv("MODEL_NAME")


# Download and Load the BGE-Small model to a local directory
local_model_path = "./bge-small-en"
snapshot_download(repo_id="BAAI/bge-small-en", local_dir=local_model_path)

# Now load the locally saved model
bge_model = SentenceTransformer(local_model_path)

# Load the BGE-Small model and store in cache
# model = SentenceTransformer("BAAI/bge-small-en")


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


def store_embeddings(data, model, index_file="bge_vector_store.index", save_index=False):
    """
    Generate embeddings using the model, store them in a FAISS index, and optionally save the index to a file.
    
    Parameters:
    data (list of str): List of text data to embed.
    model (object): Model to generate embeddings.
    index_file (str): File name to save the FAISS index.
    save_index (bool): Whether to save the FAISS index to a file (default: False).
    """
    # Generate embeddings
    embeddings = model.encode(data, normalize_embeddings=True)  # Ensure BGE models require normalization
    
    # Convert to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Save FAISS index if required
    if save_index:
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_file)
        print("Embeddings stored successfully!")
    
    return embeddings



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
    # query_embedding = np.array(query_embedding, dtype=np.float32)


    # Load FAISS index
    index = faiss.read_index(index_file_name)
    
    # Retrieve top-k similar documents
    D, I = index.search(query_embedding, top_k)
    retrieved_docs = [data[i] for i in I[0]]
    context = "\n".join(retrieved_docs)
    
    # Generate response using OpenAI API
    messages=[{"role": "system", "content": "You are an AI assistant with domain expertise."},
                  {"role": "user", "content": f"Using this information: {context},answer this question: {query}."}]
    

    response = get_chat_response(client, llm_model, messages)
    
    return response

if __name__ == "__main__":
    
    iphone_16 ="""The iPhone 16 series marks a significant leap forward in Apple's smartphone technology, showcasing a refined blend of hardware and software advancements. The series, revealed at Apple's "It's Glowtime" event in September 2024, introduces a powerful new A18 Bionic chip, designed to handle the increased computational demands of the latest features, particularly those related to artificial intelligence. Central to this advancement is the integration of iOS 18, which brings forth Apple Intelligence, a suite of AI-driven functionalities aimed at creating a more personalized and intuitive user experience. These AI capabilities manifest in various ways, from predictive text and smart suggestions during messaging to real-time language translation and intelligent email drafting. The focus on AI is evident in the device's ability to anticipate user needs, adapt to usage patterns, and streamline everyday tasks, thus enhancing productivity and convenience. Camera technology receives a substantial upgrade across the series, with particular emphasis on the Pro models. These models boast enhanced camera systems, featuring improved sensors, lenses, and image processing algorithms. The introduction of features like Camera Control provides users with greater creative flexibility, allowing for more precise adjustments and capturing higher-quality images and videos. The Pro models also feature larger displays, offering a more immersive viewing experience for multimedia consumption and productivity tasks. The iPhone 16 series comprises the standard iPhone 16 and iPhone 16 Plus, catering to a broad audience, and the premium iPhone 16 Pro and iPhone 16 Pro Max, designed for users seeking the most advanced features and performance. Additionally, Apple introduced the iPhone 16e, a more budget-friendly option, expanding the series' accessibility. This strategic diversification ensures that the iPhone 16 lineup caters to a wide range of user preferences and budgets. The global launch of the iPhone 16 series was meticulously orchestrated, with a staggered release schedule to accommodate different markets. Following the initial unveiling and pre-order phase, sales commenced in key regions on September 20, 2024, with subsequent launches in other areas like Macao and Vietnam on September 27, 2024. This phased approach allowed Apple to manage supply and demand effectively while ensuring a smooth rollout. Alongside the iPhone 16 series, Apple also introduced the Apple Watch Series 10 and AirPods 4, further expanding its ecosystem of interconnected devices. These complementary products enhance the overall Apple experience, providing seamless integration and functionality. The emphasis on AI, camera enhancements, and performance upgrades underscores Apple's commitment to innovation and user satisfaction. The iPhone 16 series represents a significant step in the evolution of smartphones, offering a compelling combination of advanced technology and user-centric design."""
    samsung_s24="""The Samsung Galaxy S24 Ultra represents a significant advancement in the premium smartphone market, showcasing Samsung's commitment to cutting-edge technology and user-centric design. Released on January 31, 2024, following its unveiling at the Galaxy Unpacked event on January 17, 2024, the S24 Ultra distinguishes itself with a suite of impressive features. At its core, the device is powered by the Snapdragon 8 Gen 3 processor, delivering exceptional performance and efficiency. A key highlight is its enhanced camera system, featuring a 200MP main sensor, along with improved telephoto capabilities, enabling users to capture stunningly detailed photos and videos. Samsung has also placed a strong emphasis on artificial intelligence, integrating Galaxy AI into various aspects of the device's functionality. This AI integration enhances everything from camera performance and image editing to real-time language translation and intelligent text summarization. The device's display is another standout feature, boasting a vibrant Dynamic AMOLED 2X screen with Corning Gorilla Armor, providing exceptional clarity and durability. The S24 Ultra also retains its signature S Pen, further enhancing productivity and creative potential. Design-wise, the S24 Ultra features a titanium frame, adding a premium touch and increased durability. The device is available in a range of sophisticated colors, including Titanium Gray, Titanium Black, Titanium Violet, and Titanium Yellow. Overall, the Samsung Galaxy S24 Ultra aims to deliver a comprehensive and premium smartphone experience, combining powerful performance, advanced camera technology, and intelligent AI features."""
    
    data = [iphone_16, samsung_s24]

    # Embeddings
    # store_embeddings(data, bge_model, save_index=True)

    user_prompt = f"What's new in iphone 16?"
    response = retrieve_and_respond(client,model_name,user_prompt,bge_model, "bge_vector_store.index", data, top_k=1)

    print(response)