import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ─── Import boat module safely ─────────────────────────────────────────────────
try:
    from boat.boat_module import predict as boat_predict
    logger.info("boat module loaded successfully")
except ImportError as e:
    logger.warning(f"boat module not found — HR chat disabled: {e}")
    boat_predict = None


class AIService:

    @staticmethod
    def chat_hr(message: str) -> dict:
        """
        Send a message through the RAG pipeline.
        Falls back to Groq LLM automatically inside boat_module when RAG can't answer.
        """
        if not boat_predict:
            return {
                'status': 'error',
                'answer': (
                    'HR assistant is currently unavailable. '
                    'Please contact administration@lsituoe.edu.pk'
                )
            }

        try:
            result = boat_predict({'action': 'ask', 'question': message})
            return result
        except Exception as e:
            logger.error(f"AIService.chat_hr error: {e}")
            return {
                'status': 'error',
                'answer': 'An error occurred processing your request. Please try again.'
            }

    @staticmethod
    def get_status() -> dict:
        """Check if HR module and knowledge base are loaded"""
        if not boat_predict:
            return {'loaded': False, 'qa_pairs': 0}
        try:
            result = boat_predict({'action': 'status'})
            return {
                'loaded': result.get('data', {}).get('documents_loaded', False),
                'qa_pairs': result.get('data', {}).get('knowledge_base_size', 0),
                'groq_available': result.get('data', {}).get('groq_available', False)
            }
        except Exception:
            return {'loaded': False, 'qa_pairs': 0}