import logging
from flask import session, jsonify, request
from datetime import datetime
from models.chat import ChatMessage, ChatSession
from models.db import db
from services.ai_service import AIService

logger = logging.getLogger(__name__)


def _generate_title(first_message: str) -> str:
    title = first_message.strip()
    return title[:60] + '…' if len(title) > 60 else title


class ChatController:

    @staticmethod
    def send_message():
        try:
            user_id = session.get('user_id') or 1
            data = request.get_json(silent=True)
            if not data:
                return jsonify({'error': 'Invalid JSON body'}), 400

            message = (data.get('message') or '').strip()
            if not message:
                return jsonify({'error': 'No message provided'}), 400

            session_id = data.get('session_id')

            if session_id:
                chat_session = ChatSession.query.filter_by(
                    id=session_id, user_id=user_id).first()
                if not chat_session:
                    return jsonify({'error': 'Session not found'}), 404
            else:
                chat_session = ChatSession(
                    user_id=user_id,
                    title=_generate_title(message)
                )
                db.session.add(chat_session)
                db.session.flush()

            user_msg = ChatMessage(
                session_id=chat_session.id,
                user_id=user_id,
                sender='user',
                message=message,
                chat_type='hr'
            )
            db.session.add(user_msg)

            result = AIService.chat_hr(message)
            answer = (
                result.get('answer') or
                'Sorry, I could not find an answer. Please contact administration@lsituoe.edu.pk'
            )

            ai_msg = ChatMessage(
                session_id=chat_session.id,
                user_id=user_id,
                sender='ai',
                message=answer,
                chat_type='hr'
            )
            db.session.add(ai_msg)
            chat_session.updated_at = datetime.utcnow()
            db.session.commit()

            return jsonify({
                'answer': answer,
                'session_id': chat_session.id,
                'session_title': chat_session.title,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"send_message error: {e}")
            db.session.rollback()
            return jsonify({
                'answer': 'An error occurred. Please try again or contact administration@lsituoe.edu.pk'
            }), 500

    @staticmethod
    def get_sessions():
        try:
            user_id = session.get('user_id') or 1
            sessions = (
                ChatSession.query
                .filter_by(user_id=user_id)
                .order_by(ChatSession.updated_at.desc())
                .limit(50)
                .all()
            )
            return jsonify({'sessions': [s.to_dict() for s in sessions]})
        except Exception as e:
            logger.error(f"get_sessions error: {e}")
            return jsonify({'sessions': []}), 500

    @staticmethod
    def get_session_messages(session_id):
        try:
            user_id = session.get('user_id') or 1
            chat_session = ChatSession.query.filter_by(
                id=session_id, user_id=user_id).first()
            if not chat_session:
                return jsonify({'error': 'Session not found'}), 404
            messages = (
                ChatMessage.query
                .filter_by(session_id=session_id)
                .order_by(ChatMessage.created_at.asc())
                .all()
            )
            return jsonify({
                'session': chat_session.to_dict(),
                'messages': [m.to_dict() for m in messages]
            })
        except Exception as e:
            logger.error(f"get_session_messages error: {e}")
            return jsonify({'messages': []}), 500

    @staticmethod
    def delete_session(session_id):
        try:
            user_id = session.get('user_id') or 1
            chat_session = ChatSession.query.filter_by(
                id=session_id, user_id=user_id).first()
            if not chat_session:
                return jsonify({'error': 'Session not found'}), 404
            db.session.delete(chat_session)
            db.session.commit()
            return jsonify({'status': 'deleted'})
        except Exception as e:
            logger.error(f"delete_session error: {e}")
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

    @staticmethod
    def get_history():
        try:
            user_id = session.get('user_id') or 1
            messages = (
                ChatMessage.query
                .filter_by(user_id=user_id, chat_type='hr')
                .order_by(ChatMessage.created_at.desc())
                .limit(50)
                .all()
            )
            return jsonify({'messages': [m.to_dict() for m in reversed(messages)]})
        except Exception as e:
            return jsonify({'messages': []}), 500