from django.apps import AppConfig
import logging
import os

logger = logging.getLogger(__name__)

class RagAgentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rag_agent"
    reasoning_service = None
    enhanced_reasoning_service = None
    turbo_reasoning_service = None

    def ready(self):
        """
        This method is called when Django starts.
        We initialize turbo services for maximum performance.
        """
        # Avoid initializing in sub-processes like manage.py makemigrations
        if os.environ.get('RUN_MAIN', None) == 'true' or "runserver" in os.sys.argv:

            # Check if turbo mode is enabled (default: enabled)
            use_turbo = os.environ.get('USE_TURBO_MODE', 'true').lower() == 'true'

            if use_turbo:
                logger.info("🚀 Initializing TurboReasoningService with FAISS acceleration...")
                try:
                    # Create cache directories
                    os.makedirs('/tmp/claude', exist_ok=True)

                    from .turbo_services import TurboReasoningService
                    self.turbo_reasoning_service = TurboReasoningService()
                    logger.info("🚀 TurboReasoningService initialized successfully!")

                except Exception as e:
                    logger.error(f"Failed to initialize TurboReasoningService: {e}", exc_info=True)
                    logger.warning("Falling back to Enhanced ReasoningService...")
                    use_turbo = False

            if not use_turbo:
                logger.info("Initializing Enhanced ReasoningService with real-time entity extraction...")
                try:
                    # Initialize both services for backward compatibility
                    from .services import ReasoningService
                    from .enhanced_services import EnhancedReasoningService

                    self.reasoning_service = ReasoningService()
                    self.enhanced_reasoning_service = EnhancedReasoningService()

                    logger.info("Enhanced ReasoningService with real-time NER initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize Enhanced ReasoningService: {e}", exc_info=True)

                    # Fallback to regular service if enhanced fails
                    try:
                        from .services import ReasoningService
                        self.reasoning_service = ReasoningService()
                        logger.warning("Falling back to regular ReasoningService")
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}", exc_info=True)
                        raise fallback_error
