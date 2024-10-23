from flask import Flask, render_template, request, jsonify
import logging
import os

from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Set Flask's log level to DEBUG
app.logger.setLevel(logging.DEBUG)

# Global variables
global chatbot_llm_chain
global knowledgebase_qa

# Function to set up the chatbot LLM chain
def setup_chatbot_llm():
    global chatbot_llm_chain
    try:
        template = """
        You are a chatbot that had a conversation with a human. Consider the previous conversation to answer the new question.

        Previous conversation: {chat_history}
        New human question: {question}

        Response:"""
        
        prompt = PromptTemplate(template=template, input_variables=["question", "chat_history"])
        llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
        memory = ConversationBufferMemory(memory_key="chat_history")
        chatbot_llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, memory=memory)
        app.logger.debug('Successfully set up chatbot_llm_chain')
    except Exception as e:
        app.logger.error(f"Error setting up chatbot_llm_chain: {e}")
        chatbot_llm_chain = None

# Function to set up the knowledgebase LLM
def setup_knowledgebase_llm():
    global knowledgebase_qa
    app.logger.debug('Setting up knowledgebase')
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        knowledgebase_qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        app.logger.debug("Successfully set up the knowledgebase")
    except Exception as e:
        app.logger.error(f"Error setting up knowledgebase LLM: {e}")
        knowledgebase_qa = None

# Combined setup function
def setup():
    setup_chatbot_llm()
    setup_knowledgebase_llm()

# Function to get an answer from the knowledgebase
def answer_from_knowledgebase(message):
    global knowledgebase_qa
    if knowledgebase_qa is None:
        app.logger.error("knowledgebase_qa is not initialized.")
        return "Knowledgebase is currently unavailable. Please try again later."
    
    app.logger.debug('Querying knowledgebase')
    try:
        res = knowledgebase_qa({"query": message})
        app.logger.debug('Query successful')
        return res['result']
    except Exception as e:
        app.logger.error(f"Error querying knowledgebase: {e}")
        return "Failed to retrieve answer from the knowledgebase."

# Function to search the knowledgebase
def search_knowledgebase(message):
    global knowledgebase_qa
    if knowledgebase_qa is None:
        app.logger.error("knowledgebase_qa is not initialized.")
        return "Knowledgebase is currently unavailable."
    
    app.logger.debug('Searching knowledgebase')
    try:
        res = knowledgebase_qa({"query": message})
        sources = ""
        for count, source in enumerate(res['source_documents'], 1):
            sources += f"Source {count}\n{source.page_content}\n"
        return sources
    except Exception as e:
        app.logger.error(f"Error searching knowledgebase: {e}")
        return "Failed to search the knowledgebase."

# Function to get an answer from the chatbot
def answer_as_chatbot(message):
    global chatbot_llm_chain
    if chatbot_llm_chain is None:
        app.logger.error("chatbot_llm_chain is not initialized.")
        return "Chatbot is currently unavailable. Please try again later."
    
    try:
        res = chatbot_llm_chain.run(message)
        return res
    except Exception as e:
        app.logger.error(f"Error generating chatbot response: {e}")
        return "Failed to get a response from the chatbot."

# Route to handle knowledgebase answers
@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json['message']
    response_message = answer_from_knowledgebase(message)
    return jsonify({'message': response_message}), 200

# Route to handle knowledgebase search
@app.route('/search', methods=['POST'])
def search():
    message = request.json['message']
    response_message = search_knowledgebase(message)
    return jsonify({'message': response_message}), 200

# Route to handle chatbot responses
@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    response_message = answer_as_chatbot(message)
    return jsonify({'message': response_message}), 200

# Main index route
@app.route("/")
def index():
    return render_template("index.html", title="")

# Main function to start the app
if __name__ == "__main__":
    setup()
    app.run(host='0.0.0.0', port=5001, debug=True)
