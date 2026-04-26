import os
import requests
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────────
CONFIG = {
    'min_similarity_threshold': 0.30,
    'max_context_docs': 3,
}

# ── HuggingFace Inference API for embeddings ────────────────────────────────────
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    if not HF_API_KEY:
        logger.warning("No HuggingFace API key set")
        return None
    try:
        response = requests.post(
            HF_EMBED_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            # HF returns list of lists
            return np.array(result)
        logger.error(f"HF embed error: {response.status_code} {response.text}")
        return None
    except Exception as e:
        logger.error(f"HF embed exception: {e}")
        return None

# ── Groq LLM fallback (free) ───────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def llm_fallback(question: str, context: str = "") -> str:
    if not GROQ_API_KEY:
        return (
            "I couldn't find a specific answer. "
            "Please contact administration@lsituoe.edu.pk for assistance."
        )
    try:
        system_prompt = (
            "You are a professional HR assistant for LSIT University of Education, Pakistan. "
            "Answer questions about HR policies, attendance, leave, payroll, and career development. "
            "Be concise (2-4 sentences max), professional, and helpful. "
            "If unsure about LSIT-specific details, say so and suggest contacting administration@lsituoe.edu.pk."
        )
        user_content = (
            f"Use this HR policy context if relevant:\n\n{context}\n\nQuestion: {question}"
            if context else question
        )
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 250,
                "temperature": 0.2
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        logger.error(f"Groq API error: {response.status_code} {response.text}")
        return "I couldn't find a specific answer. Please contact administration@lsituoe.edu.pk"
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again or contact administration@lsituoe.edu.pk"
    except Exception as e:
        logger.error(f"LLM fallback error: {e}")
        return "I encountered an issue. Please contact administration@lsituoe.edu.pk"


# ── Load HR documents ──────────────────────────────────────────────────────────
docs_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hr_docs")
knowledge_base = []
knowledge_metadata = []
embeddings = None


def smart_chunk_text(text: str, filename: str) -> List[str]:
    chunks = []
    text = text.strip()
    if not text:
        return chunks
    qa_pattern = r'(Q:.*?\nA:.*?)(?=\nQ:|$)'
    qa_matches = re.findall(qa_pattern, text, re.DOTALL | re.IGNORECASE)
    if qa_matches:
        for match in qa_matches:
            clean_match = match.strip()
            if len(clean_match) > 20:
                chunks.append(clean_match)
        return chunks[:20]
    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        clean_para = para.strip()
        if len(clean_para) > 20:
            chunks.append(clean_para)
    if not chunks and len(text) > 20:
        chunks.append(text)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash(chunk[:100])
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique_chunks.append(chunk)
    return unique_chunks[:20]


def extract_keywords(text: str) -> set:
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'be'}
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {word for word in words if word not in stopwords}


if not os.path.exists(docs_folder):
    print(f"ERROR: HR documents folder '{docs_folder}' not found!")
else:
    print(f"Loading HR documents from: {docs_folder}")
    for filename in os.listdir(docs_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_folder, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    if not text.strip():
                        print(f"  Warning: {filename} is empty, skipping...")
                        continue
                    chunks = smart_chunk_text(text, filename)
                    for chunk in chunks:
                        keywords = extract_keywords(chunk)
                        knowledge_base.append(chunk)
                        knowledge_metadata.append({
                            'source': filename,
                            'keywords': keywords,
                            'preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                            'length': len(chunk),
                            'full_text': chunk
                        })
                    print(f"  Loaded {len(chunks)} chunks from {filename}")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
    print(f"Total: {len(knowledge_base)} knowledge chunks loaded")
    if knowledge_base and HF_API_KEY:
        print("Creating embeddings via HuggingFace API...")
        embeddings = get_embeddings(knowledge_base)
        if embeddings is not None:
            print("Embeddings ready!")
        else:
            print("Warning: embeddings failed, RAG disabled — keyword matching still active")


# ── Search ─────────────────────────────────────────────────────────────────────
def search_docs(query: str, top_k: int = 3) -> List[Dict]:
    if not knowledge_base or embeddings is None:
        return []
    query_emb = get_embeddings([query])
    if query_emb is None:
        return []
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_k_actual = min(top_k * 3, len(sims))
    top_indices = sims.argsort()[-top_k_actual:][::-1]
    query_keywords = {w for w in query.lower().split() if len(w) > 2}
    relevant_docs = []
    for idx in top_indices:
        keyword_boost = 0
        if query_keywords and 'keywords' in knowledge_metadata[idx]:
            keyword_match = len(query_keywords & knowledge_metadata[idx]['keywords'])
            keyword_boost = keyword_match * 0.1
        final_score = sims[idx] + keyword_boost
        if final_score >= CONFIG['min_similarity_threshold']:
            relevant_docs.append({
                'content': knowledge_base[idx],
                'score': final_score,
                'source': knowledge_metadata[idx]['source'],
                'full_text': knowledge_metadata[idx]['full_text']
            })
    relevant_docs.sort(key=lambda x: x['score'], reverse=True)
    return relevant_docs[:top_k]


# ── Extract best answer from RAG ───────────────────────────────────────────────
def get_best_answer_from_context(question: str, context_docs: List[Dict]) -> Optional[str]:
    if not context_docs:
        return None

    question_lower = question.lower()
    common_words = {
        'where', 'what', 'how', 'do', 'i', 'my', 'the', 'a', 'an', 'is',
        'are', 'to', 'for', 'can', 'you', 'help', 'me', 'please', 'am',
        'would', 'like', 'need', 'want', 'does', 'have', 'we', 'our'
    }
    question_words = {w for w in question_lower.split() if len(w) > 2} - common_words

    best_answer = None
    best_score = 0

    for doc in context_docs:
        content = doc['content']
        source = doc.get('source', '').lower()

        source_boost = 0
        if 'payroll' in source and any(k in question_lower for k in ['salary', 'paid', 'bank', 'pay']):
            source_boost = 0.3
        if 'attendance' in source and any(k in question_lower for k in ['attendance', 'check', 'mark', 'attendence']):
            source_boost = 0.3
        if 'leave' in source and any(k in question_lower for k in ['leave', 'vacation', 'holiday']):
            source_boost = 0.3

        qa_pattern = r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|$)'
        qa_matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)

        for q, a in qa_matches:
            q_lower = q.lower()
            a_clean = a.strip()
            q_words = {w for w in q_lower.split() if len(w) > 2} - common_words

            word_score = (
                len(question_words & q_words) / max(len(question_words), 1)
                if question_words else 0
            )
            keyword_score = 0
            for user_kw, doc_kws in [
                ('salary', ['salary', 'pay']),
                ('history', ['history']),
                ('bank', ['bank', 'account']),
                ('update', ['update', 'change']),
                ('attend', ['attend', 'mark']),
                ('attendence', ['attend', 'mark']),
                ('leave', ['leave', 'vacation']),
                ('check', ['check', 'login']),
                ('slip', ['slip', 'payroll']),
                ('increment', ['increment', 'raise']),
                ('training', ['training', 'course']),
            ]:
                if user_kw in question_lower and any(k in q_lower for k in doc_kws):
                    keyword_score += 0.4

            total_score = word_score + keyword_score + source_boost
            if total_score > best_score:
                best_score = total_score
                best_answer = a_clean

        if best_answer and best_score > 0.3:
            return best_answer

    return best_answer


# ── Greetings ──────────────────────────────────────────────────────────────────
def get_simple_response(question: str) -> Optional[str]:
    q = question.lower().strip()
    greetings = ["hi", "hello", "hey", "hii", "helloo", "greetings",
                 "good morning", "good afternoon", "good evening", "salam",
                 "assalam", "assalamualaikum"]
    acknowledgments = ["ok", "okay", "thanks", "thank you", "got it",
                       "sure", "fine", "cool", "nice", "alright", "shukria"]
    if any(q == g or q.startswith(g + " ") for g in greetings):
        return (
            "Hello! I'm the LSIT HR Assistant. I can help you with:\n"
            "• Attendance — check-in/out, missed attendance\n"
            "• Leave — how to apply, leave balance\n"
            "• Payroll — salary dates, salary slips\n"
            "• Career Development — training programs\n\n"
            "What would you like to know?"
        )
    if any(ack in q for ack in acknowledgments):
        return "You're welcome! Feel free to ask anything else about LSIT HR policies."
    return None


# ── Flexible keyword matching ──────────────────────────────────────────────────
def match_quick_answer(q_lower: str) -> Optional[str]:
    # Direct substring match first
    for key, response in QUICK_ANSWERS.items():
        if key in q_lower:
            return response

    # Flexible word overlap match
    q_words = set(q_lower.split())
    best_match = None
    best_overlap = 0
    for key, response in QUICK_ANSWERS.items():
        key_words = set(key.split())
        overlap = len(key_words & q_words)
        if overlap >= max(1, len(key_words) - 1) and overlap > best_overlap:
            best_overlap = overlap
            best_match = response

    return best_match


# ── Quick keyword answers ──────────────────────────────────────────────────────
QUICK_ANSWERS = {
    # Attendance
    "check in": "Click 'Check In' on the LSIT HR portal (https://lsit.edu.pk/attendance) or LSIT Connect app before 10:00 AM daily. After 10:00 AM it is marked as late.",
    "check out": "Click 'Check Out' at end of day on the LSIT HR portal or LSIT Connect app. Required for accurate attendance records.",
    "mark attendance": "Log in to https://lsit.edu.pk/attendance or use the LSIT Connect mobile app. Click Check In at start and Check Out at end of day.",
    "mark attendence": "Log in to https://lsit.edu.pk/attendance or use the LSIT Connect mobile app. Click Check In at start and Check Out at end of day.",
    "where attendance": "Log in to https://lsit.edu.pk/attendance or use the LSIT Connect mobile app. Click Check In at start and Check Out at end of day.",
    "attendance portal": "Access attendance at https://lsit.edu.pk/attendance using your LSIT credentials.",
    "forgot attendance": "Contact administration@lsituoe.edu.pk within 24 hours with your employee ID and date to get attendance rectified.",
    "missed attendance": "Contact administration@lsituoe.edu.pk within 24 hours with your employee ID and date to get attendance rectified.",
    "late check in": "Any check-in after 10:00 AM is marked as late in your attendance record.",
    "attendance system": "Use the LSIT HR portal at https://lsit.edu.pk/attendance or the LSIT Connect mobile app for all attendance needs.",

    # Leave
    "leave balance": "Check your leave balance in the LSIT HR portal under 'Leave Balance'. For queries: hr@lsituoe.edu.pk",
    "check leave": "Check your leave balance in the LSIT HR portal under 'Leave Balance'. For queries: hr@lsituoe.edu.pk",
    "apply leave": "Log into LSIT HR portal → Leave Requests → fill the form with dates and reason → submit. Planned leave needs 3 days advance notice.",
    "leave apply": "Log into LSIT HR portal → Leave Requests → fill the form with dates and reason → submit. Planned leave needs 3 days advance notice.",
    "how to apply leave": "Log into LSIT HR portal → Leave Requests → fill the form with dates and reason → submit. Planned leave needs 3 days advance notice.",
    "annual leave": "You get 20 days annual leave per year.",
    "sick leave": "You get 10 days sick leave per year. Notify your manager by 9:00 AM on the day of absence.",
    "casual leave": "You get 5 days casual leave per year for urgent personal matters.",
    "leave types": "LSIT offers: Annual Leave (20 days), Sick Leave (10 days), and Casual Leave (5 days) per year.",
    "how many leaves": "LSIT offers: Annual Leave (20 days), Sick Leave (10 days), and Casual Leave (5 days) per year.",

    # Salary / Payroll
    "when is salary": "Salary is credited to your bank account between the 5th and 7th of each month.",
    "salary date": "Salary is processed between 5th-7th of each month. If this falls on a weekend/holiday, processed on the previous working day.",
    "salary day": "Salary is credited to your bank account between the 5th and 7th of each month.",
    "salary slip": "Download your salary slip from LSIT HR portal under 'Payroll History', or check your email — sent monthly.",
    "payslip": "Download your salary slip from LSIT HR portal under 'Payroll History', or check your email — sent monthly.",
    "salary not received": "Contact accounts@lsituoe.edu.pk immediately with your employee ID and the month.",
    "salary not credited": "Contact accounts@lsituoe.edu.pk immediately with your employee ID and the month.",
    "update bank": "Submit new bank details to accounts department with proof, or update in LSIT HR portal under My Profile > Bank Details.",
    "bank account": "Update your bank details in LSIT HR portal under My Profile > Bank Details, or submit proof to accounts department.",
    "provident fund": "Provident Fund: you contribute 10% of basic salary, LSIT matches 10%. PF statements available quarterly.",
    "increment": "Annual performance-based increment given in July, ranging from 5% to 15%.",
    "salary increment": "Annual performance-based increment given in July, ranging from 5% to 15%.",

    # Career & Training
    "training": "LSIT offers: AWS/Azure/Google Cloud certifications, Leadership Development, Soft Skills Workshops, and Mentorship Program.",
    "career": "LSIT has three career tracks: Individual Contributor (technical), Management (leadership), and Technical Expert.",
    "career development": "LSIT has three career tracks: Individual Contributor (technical), Management (leadership), and Technical Expert.",
    "courses available": "LSIT offers: AWS/Azure/Google Cloud certifications, Leadership Development, Soft Skills Workshops, and Mentorship Program.",

    # Contact
    "contact hr": "HR: hr@lsituoe.edu.pk | Administration: administration@lsituoe.edu.pk | Payroll: accounts@lsituoe.edu.pk",
    "hr email": "HR: hr@lsituoe.edu.pk | Administration: administration@lsituoe.edu.pk | Payroll: accounts@lsituoe.edu.pk",
    "hr contact": "HR: hr@lsituoe.edu.pk | Administration: administration@lsituoe.edu.pk | Payroll: accounts@lsituoe.edu.pk",
}


# ── Main answer generator ──────────────────────────────────────────────────────
def generate_answer(question: str) -> str:
    q_lower = question.lower().strip()

    # 1. Greeting / acknowledgment
    simple = get_simple_response(question)
    if simple:
        return simple

    # 2. Flexible quick keyword match
    quick = match_quick_answer(q_lower)
    if quick:
        return quick

    # 3. No knowledge base — go straight to LLM
    if not knowledge_base:
        return llm_fallback(question)

    # 4. RAG semantic search
    relevant_docs = search_docs(question, top_k=3)

    if relevant_docs:
        best_answer = get_best_answer_from_context(question, relevant_docs)
        if best_answer:
            best_answer = best_answer.strip()
            if best_answer.startswith('Q:'):
                best_answer = best_answer.split('A:')[-1].strip()
            return best_answer
        # RAG found docs but no clean answer — use LLM with context
        context = "\n\n".join([d['content'] for d in relevant_docs[:2]])
        return llm_fallback(question, context)

    # 5. Nothing matched — LLM from general HR knowledge
    return llm_fallback(question)


# ── Stats ──────────────────────────────────────────────────────────────────────
def get_document_stats() -> Dict:
    sources = {}
    for meta in knowledge_metadata:
        source = meta['source']
        if source not in sources:
            sources[source] = {'chunks': 0, 'total_chars': 0}
        sources[source]['chunks'] += 1
        sources[source]['total_chars'] += meta.get('length', 0)
    return {
        'total_chunks': len(knowledge_base),
        'total_documents': len(sources),
        'documents': sources
    }


# ── Flask predict entry point ──────────────────────────────────────────────────
def predict(data: dict) -> dict:
    action = data.get('action')

    if action == 'ask':
        question = data.get('question', '').strip()
        if not question:
            return {'status': 'error', 'answer': 'No question provided.'}
        try:
            start_time = datetime.now()
            answer = generate_answer(question)
            elapsed = (datetime.now() - start_time).total_seconds()
            return {'status': 'success', 'answer': answer, 'response_time': round(elapsed, 2)}
        except Exception as e:
            logger.error(f"predict error: {e}")
            return {'status': 'error', 'answer': f"Error: {str(e)}"}

    elif action == 'search':
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        if not query:
            return {'status': 'error', 'results': []}
        try:
            results = search_docs(query, top_k)
            clean = [{'content': r['content'][:300], 'score': round(float(r['score']), 3),
                      'source': r['source']} for r in results]
            return {'status': 'success', 'results': clean}
        except Exception as e:
            return {'status': 'error', 'results': [], 'error': str(e)}

    elif action == 'status':
        return {'status': 'success', 'data': {
            'knowledge_base_size': len(knowledge_base),
            'documents_loaded': len(knowledge_base) > 0,
            'groq_available': bool(GROQ_API_KEY),
            'embeddings_ready': embeddings is not None,
        }}

    elif action == 'list_documents':
        stats = get_document_stats()
        return {'status': 'success', 'documents': list(stats['documents'].keys())}

    else:
        return {'status': 'error', 'answer': 'Unknown action.'}


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("LSIT HR Assistant (RAG + Groq fallback)")
    print("=" * 50)
    print(f"Loaded: {len(knowledge_base)} chunks | Groq: {'enabled' if GROQ_API_KEY else 'disabled'} | Embeddings: {'ready' if embeddings is not None else 'disabled'}")
    print("Type 'exit' to quit\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break
        if not question:
            continue
        start = datetime.now()
        answer = generate_answer(question)
        elapsed = (datetime.now() - start).total_seconds()
        print(f"Bot: {answer}")
        print(f"({elapsed:.2f}s)\n")