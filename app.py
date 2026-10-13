from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file, session, redirect, url_for, g
import sys
import os
import re
import time
import json
import uuid
import sqlite3
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helper import download_embeddings, extract_data_from_pdf_directory, optimized_text_splitter
from src.prompt import system_prompt
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate


# Updated / Correct:
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
# Initialize Flask app
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'civic-chatbot-secret-key-2026')

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['AUDIO_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'audio')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Database Setup
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'civic.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

init_db()

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=env_path, override=True)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Memory storage
conversation_memories = {}

class SessionMemoryManager:
    def __init__(self, memory_type="buffer", max_token_limit=2000, k=10):
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.k = k
        self.last_activity = datetime.now()
        
        if memory_type == "window":
            self.memory = ConversationBufferWindowMemory(
                k=k,
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
        else:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
    
    def add_message(self, human_message, ai_message):
        self.memory.save_context(
            {"input": human_message},
            {"answer": ai_message}
        )
        self.last_activity = datetime.now()
    
    def get_memory_variables(self):
        return self.memory.load_memory_variables({})
    
    def get_chat_history(self):
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        formatted_history = []
        for message in chat_history:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")
        return "\n".join(formatted_history)
    
    def clear_memory(self):
        self.memory.clear()
        self.last_activity = datetime.now()
    
    def get_message_count(self):
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        return len(chat_history)

def get_or_create_memory(session_id, memory_type="buffer", max_token_limit=2000, k=10):
    if session_id not in conversation_memories:
        conversation_memories[session_id] = SessionMemoryManager(
            memory_type=memory_type,
            max_token_limit=max_token_limit,
            k=k
        )
    return conversation_memories[session_id]

def cleanup_old_memories():
    cutoff_time = datetime.now() - timedelta(hours=24)
    to_remove = [sid for sid, m in conversation_memories.items() if m.last_activity < cutoff_time]
    for sid in to_remove:
        del conversation_memories[sid]

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""{system_prompt}

Previous conversation history:
{{chat_history}}

Use the conversation history to provide contextual responses and maintain continuity.
If the user refers to something mentioned earlier, acknowledge it appropriately."""),
    ("human", "{input}"),
])

def create_conversational_rag_chain(retriever, llm, prompt):
    def _combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _format_chat_history(chat_history):
        if not chat_history:
            return ""
        formatted = []
        for message in chat_history:
            if isinstance(message, HumanMessage):
                formatted.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted.append(f"Assistant: {message.content}")
        return "\n".join(formatted)
    
    def run_chain(input_data):
        docs = retriever.invoke(input_data["input"])
        chat_history = _format_chat_history(input_data.get("chat_history", []))
        formatted_prompt = prompt.format(
            input=input_data["input"],
            chat_history=chat_history,
            context=_combine_docs(docs)
        )
        response = llm.invoke(formatted_prompt)
        return {
            "answer": response.content,
            "source_documents": docs
        }
    
    return run_chain

conversational_rag_chain = None
retriever = None
llm = None

def init_rag():
    global conversational_rag_chain, retriever, llm
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'), override=True)
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    if pinecone_key:
        os.environ["PINECONE_API_KEY"] = pinecone_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if not pinecone_key or not groq_key:
        print("Note: PINECONE_API_KEY or GROQ_API_KEY is not set. Add them to .env to enable RAG responses.")
        return None
    try:
        pc = Pinecone(api_key=pinecone_key)
        index_name = "chatbot"
        embeddings = download_embeddings()

        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            namespace="default"
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 12})

        model_name = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
        llm = ChatOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            model=model_name,
            temperature=0.3,
            max_completion_tokens=3050,
            top_p=0.95,
            stop=None,
        )

        conversational_rag_chain = create_conversational_rag_chain(retriever, llm, prompt)
        print("RAG chain initialized successfully.")
        return conversational_rag_chain
    except Exception as e:
        print(f"Error initializing RAG chain: {e}")
        return None

init_rag()

# Format and clean chatbot response into proper English
def clean_response(text):
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```[a-zA-Z]*\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s*(.*?)$", r"\1:", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\*{2,}([^*\n]+)\*{2,}", r"\1", text)
    text = re.sub(r"_{2,}([^_\n]+)_{2,}", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)
    text = re.sub(r"(\*{2,}|#{2,}|---|___)", "", text)
    text = re.sub(r"\s*(Answer:|Legal Basis:|Context:|Follow-up:)\s*", r"\n\n\1\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Authentication APIs
@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.get_json(silent=True) or request.form.to_dict()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify(status='error', message='Email and password are required.'), 400
    if len(password) < 6:
        return jsonify(status='error', message='Password must be at least 6 characters.'), 400
    if not re.fullmatch(r'[^@]+@[^@]+\.[^@]+', email):
        return jsonify(status='error', message='Invalid email address.'), 400
        
    db = get_db()
    existing = db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
    if existing:
        return jsonify(status='error', message='An account with this email already exists.'), 409
        
    user_id = str(uuid.uuid4())
    pw_hash = generate_password_hash(password)
    db.execute('INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)', (user_id, email, pw_hash))
    db.commit()
    
    session['user_id'] = user_id
    session['user_email'] = email
    return jsonify(status='success', user={'id': user_id, 'email': email, 'displayName': email.split('@')[0]})

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True) or request.form.to_dict()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify(status='error', message='Email and password are required.'), 400
        
    db = get_db()
    row = db.execute('SELECT id, email, password_hash FROM users WHERE email = ?', (email,)).fetchone()
    if not row or not check_password_hash(row['password_hash'], password):
        return jsonify(status='error', message='Incorrect email or password.'), 401
        
    session['user_id'] = row['id']
    session['user_email'] = row['email']
    return jsonify(status='success', user={'id': row['id'], 'email': row['email'], 'displayName': row['email'].split('@')[0]})

@app.route('/api/auth/logout', methods=['POST', 'GET'])
def api_logout():
    session.pop('user_id', None)
    session.pop('user_email', None)
    return jsonify(status='success')

@app.route('/api/auth/user', methods=['GET'])
def api_user():
    if 'user_id' in session:
        email = session.get('user_email', '')
        return jsonify(authenticated=True, user={'id': session['user_id'], 'email': email, 'displayName': email.split('@')[0]})
    return jsonify(authenticated=False)

# Web Page Routes
@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    cleanup_old_memories()
    return render_template('landing.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/lawyer_portal')
def lawyer_portal():
    return render_template('lawyer_portal.html')

@app.route('/lawyer', methods=['GET', 'POST'])
def lawyer():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        area = request.form.get('area')
        details = request.form.get('details')
        print(f"Lawyer consultation inquiry - Name: {name}, Email: {email}, Area: {area}, Details: {details}")
        return redirect(url_for('home'))
    return render_template('lawyer.html')

@app.route('/appointments')
def appointments():
    return render_template('appointments.html')

@app.route('/mycases')
def mycases():
    return render_template('mycases.html')

@app.route('/cv')
def cv():
    return render_template('cv.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/case_study')
def case_study():
    return render_template('case_study.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/profileset')
def profileset():
    return render_template('profileset.html')

# Chat response endpoints
@app.route('/get_response', methods=['POST'])
def get_chat_response():
    msg = request.form.get('msg', '')
    response_type = request.form.get('response_type', 'normal')
    memory_type = request.form.get('memory_type', 'buffer')
    
    session_id = session.get('session_id', str(uuid.uuid4()))
    if 'session_id' not in session:
        session['session_id'] = session_id
    
    memory_manager = get_or_create_memory(session_id, memory_type=memory_type)
    
    input_message = msg
    if response_type == "short":
        input_message = f"{msg} (Please provide a very brief response)"
    elif response_type == "detailed":
        input_message = f"{msg} (Please provide a detailed explanation)"
    
    try:
        memory_vars = memory_manager.get_memory_variables()
        chain_input = {
            "input": input_message,
            "chat_history": memory_vars.get("chat_history", [])
        }
        
        if conversational_rag_chain is None:
            init_rag()
        if conversational_rag_chain is None:
            return "Chatbot is running! To enable AI responses, please add valid PINECONE_API_KEY and GROQ_API_KEY to your .env file."

        response = conversational_rag_chain(chain_input)
        answer = clean_response(response["answer"])
        memory_manager.add_message(msg, answer)
        return answer
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

@app.route('/stream_response', methods=['POST'])
def stream_chat():
    msg = request.form.get('msg', '')
    response_type = request.form.get('response_type', 'normal')
    memory_type = request.form.get('memory_type', 'buffer')
    
    session_id = session.get('session_id', str(uuid.uuid4()))
    if 'session_id' not in session:
        session['session_id'] = session_id
    
    memory_manager = get_or_create_memory(session_id, memory_type=memory_type)
    
    input_message = msg
    if response_type == "short":
        input_message = f"{msg} (Please provide a very brief response)"
    elif response_type == "detailed":
        input_message = f"{msg} (Please provide a detailed explanation)"
    
    def generate():
        try:
            memory_vars = memory_manager.get_memory_variables()
            chain_input = {
                "input": input_message,
                "chat_history": memory_vars.get("chat_history", [])
            }
            
            if conversational_rag_chain is None:
                init_rag()
            if conversational_rag_chain is None:
                yield "Chatbot is running! To enable AI responses, please add valid PINECONE_API_KEY and GROQ_API_KEY to your .env file."
                return

            response = conversational_rag_chain(chain_input)
            answer = clean_response(response["answer"])
            memory_manager.add_message(msg, answer)
            
            for word in answer.split():
                yield word + " "
                time.sleep(0.04)
        except Exception as e:
            print(f"Error in stream_chat: {str(e)}")
            yield "I apologize, but I encountered an error processing your request."
    
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/clear_memory', methods=['POST'])
@app.route('/clear_file_memory', methods=['POST'])
def clear_memory():
    session_id = session.get('session_id')
    if session_id and session_id in conversation_memories:
        conversation_memories[session_id].clear_memory()
        return jsonify(status="success", message="Memory cleared")
    return jsonify(status="success", message="Memory cleared")

@app.route('/get_memory', methods=['GET'])
@app.route('/get_file_memory', methods=['GET'])
def get_memory():
    session_id = session.get('session_id')
    if session_id and session_id in conversation_memories:
        memory_manager = conversation_memories[session_id]
        chat_history = memory_manager.get_chat_history()
        message_count = memory_manager.get_message_count()
        return jsonify(status="success", chat_history=chat_history, message_count=message_count, memory_type=memory_manager.memory_type)
    return jsonify(status="success", chat_history="", message_count=0, memory_type="buffer")

@app.route('/set_memory_type', methods=['POST'])
def set_memory_type():
    session_id = session.get('session_id')
    memory_type = request.form.get("memory_type", "buffer")
    k = int(request.form.get("k", 10))
    if memory_type not in ["buffer", "window"]:
        return jsonify(status="error", message="Invalid memory type.")
    if session_id:
        conversation_memories[session_id] = SessionMemoryManager(memory_type=memory_type, k=k)
        return jsonify(status="success", message=f"Memory type set to {memory_type}", k=k if memory_type == "window" else None)
    return jsonify(status="error", message="No active session found")

@app.route('/get_memory_info', methods=['GET'])
def get_memory_info():
    session_id = session.get('session_id')
    if session_id and session_id in conversation_memories:
        memory_manager = conversation_memories[session_id]
        return jsonify(status="success", memory_type=memory_manager.memory_type, message_count=memory_manager.get_message_count(), k=memory_manager.k if memory_manager.memory_type == "window" else None, max_token_limit=memory_manager.max_token_limit)
    return jsonify(status="success", memory_type="buffer", message_count=0)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
