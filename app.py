from flask import Flask, jsonify, session, render_template
from models.db import db
from models.chat import ChatMessage, ChatSession
from controllers.chat_controller import ChatController

app = Flask(__name__)
app.secret_key = 'lsit-hr-secret-key-change-in-production'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lsit_hr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# ── Routes ──────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

# Chat
@app.route('/api/chat', methods=['POST'])
def chat():
    return ChatController.send_message()

# Sessions list
@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    return ChatController.get_sessions()

# Messages in one session
@app.route('/api/sessions/<int:session_id>', methods=['GET'])
def get_session_messages(session_id):
    return ChatController.get_session_messages(session_id)

# Delete a session
@app.route('/api/sessions/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    return ChatController.delete_session(session_id)

# Health check
@app.route('/api/health', methods=['GET'])
def health():
    try:
        from boat.boat_module import knowledge_base
        qa_count = len(knowledge_base)
    except Exception:
        qa_count = 0
    return jsonify({'status': 'ok', 'qa_pairs': qa_count})

# Legacy history
@app.route('/history', methods=['GET'])
def history():
    return ChatController.get_history()

# ── Error handlers ───────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': 'Route not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'status': 'error', 'message': 'Method not allowed'}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ── Init ──────────────────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, port=5000)