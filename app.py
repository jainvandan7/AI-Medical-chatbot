from flask import Flask, jsonify, request, render_template, session
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
import time  # For simulating delay for typing indicator
import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key"  # needed for session storage

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Initialize components
index_name = "medichatbot"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Updated system prompt for more natural conversation
system_prompt = (
    "You are Dr. Insight, a friendly medical AI assistant. Respond conversationally while "
    "providing accurate medical information. Keep answers concise (3-5 sentences) but detailed. "
    "For emotional statements, acknowledge first, then provide information if requested. "
    "Use simple language and examples when explaining medical concepts.\n\n"
    "Context:\n{context}\n\n"
    "Conversation History:\n{history}\n\n"
    "User: {input}"
)

# Initialize Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize Mistral LLM
llm = ChatMistralAI(
    model="mistral-medium",
    temperature=0.4,
    api_key=MISTRAL_API_KEY
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([ 
    ("system", system_prompt),
])

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Enhanced emotional responses
EMOTIONAL_RESPONSES = {
    "hi": "Hello! I'm Dr. Insight, your AI medical assistant. How can I help you today?",
    "hello": "Hi there! What medical questions can I answer for you?",
    "hey": "Hey! I'm here to help with health-related questions.",
    "how are you": "I'm just a program without feelings, but I'm ready to assist you!",
    "i am happy": "That's great to hear! Would you like me to explain the medical reasons behind happiness?",
    "i am sad": "I'm sorry you're feeling this way. Would you like information about mood improvement?",
    "i am stressed": "Stress can be challenging. Would you like some stress management tips?",
    "i am anxious": "Anxiety can be difficult. I can explain some coping techniques if you'd like.",
    "thank you": "You're welcome! Feel free to ask more questions anytime.",
    "thanks": "My pleasure! Let me know if you need anything else."
}

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip().lower()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    print("User query:", user_message)

    # Initialize history if not present
    if "history" not in session:
        session["history"] = []

    # 1. Handle pending emotional explanation if user says "sure", "yes", etc.
    confirmation_phrases = ["sure", "yes", "okay", "ok", "yeah", "yup"]
    if any(phrase == user_message for phrase in confirmation_phrases):
        pending_topic = session.get("pending_topic")
        if pending_topic:
            if pending_topic == "happiness":
                bot_response = (
                    "Happiness is linked to the release of chemicals like dopamine, serotonin, oxytocin, and endorphins. "
                    "These neurotransmitters promote feelings of pleasure, bonding, and well-being. "
                    "Activities like exercising, socializing, and even sunlight exposure can naturally boost them!"
                )
            elif pending_topic == "sadness":
                bot_response = (
                    "Sadness often involves lower levels of serotonin and dopamine in the brain. "
                    "Maintaining social connections, physical activity, and sometimes therapy can help restore balance."
                )
            elif pending_topic == "stress":
                bot_response = (
                    "During stress, your body releases cortisol, the 'stress hormone.' "
                    "While helpful short-term, too much cortisol can harm health. "
                    "Techniques like deep breathing, exercise, and mindfulness help manage stress."
                )
            elif pending_topic == "anxiety":
                bot_response = (
                    "Anxiety activates the brain's 'fight or flight' system. "
                    "Relaxation exercises, controlled breathing, and therapy are often effective ways to cope."
                )
            else:
                bot_response = "I'm here to help! Could you tell me a bit more about how you're feeling?"

            # Clear pending topic once answered
            session.pop("pending_topic", None)

            # Update history
            session["history"].append(f"User: {user_message}")
            session["history"].append(f"Bot: {bot_response}")

            print("Bot Response:", bot_response)
            return jsonify({"response": bot_response})

    # 2. Check if user_message matches any emotional keywords
    for keyword, emotional_reply in EMOTIONAL_RESPONSES.items():
        if keyword in user_message:
            # Save pending topic if offering explanation
            if "would you like me to explain" in emotional_reply.lower() or "would you like information" in emotional_reply.lower():
                if "happy" in keyword:
                    session["pending_topic"] = "happiness"
                elif "sad" in keyword:
                    session["pending_topic"] = "sadness"
                elif "stress" in keyword:
                    session["pending_topic"] = "stress"
                elif "anxious" in keyword:
                    session["pending_topic"] = "anxiety"
            return jsonify({"response": emotional_reply})

    # Combine past conversation history
    conversation_history = "\n".join(session["history"][-6:])

    # Send typing indicator
    time.sleep(1.5)

    try:
        # Prepare the full input for RAG
        response = rag_chain.invoke({
            "input": user_message,
            "history": conversation_history
        })

        bot_response = response.get("answer", "").strip()
        
        # Enhance the response format if it's too clinical
        if not any(bot_response.startswith(phrase) for phrase in ["That's", "I'm", "You"]):
            if "?" in user_message:
                bot_response = f"Regarding your question, {bot_response[0].lower() + bot_response[1:]}"
            else:
                bot_response = f"I understand you're asking about {user_message}. {bot_response}"

        # Final fallback if response is empty
        if not bot_response:
            bot_response = "I want to make sure I understand correctly. Could you rephrase your question?"

    except Exception as e:
        print("Error generating response:", e)
        bot_response = "I'm having trouble with that request. Could you try asking differently?"

    # Update the history
    session["history"].append(f"User: {user_message}")
    session["history"].append(f"Bot: {bot_response}")

    print("Bot Response:", bot_response)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
