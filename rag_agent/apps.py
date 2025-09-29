from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class RagAgentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rag_agent"
    reasoning_service = None

    def ready(self):
        """
        This method is called when Django starts.
        We initialize our reasoning service as a singleton here.
        """
        # Avoid initializing in sub-processes like manage.py makemigrations
        import os
        if os.environ.get('RUN_MAIN', None) == 'true' or "runserver" in os.sys.argv:
            logger.info("Initializing ReasoningService...")
            try:
                from .services import ReasoningService
                self.reasoning_service = ReasoningService()
                logger.info("ReasoningService initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize ReasoningService: {e}", exc_info=True)
                # Depending on the desired behavior, you might want to raise the exception
                # to prevent the server from starting with a non-functional service.
                raise e
