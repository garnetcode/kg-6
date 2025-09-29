from django.shortcuts import render
from django.apps import apps
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import time
import logging

logger = logging.getLogger(__name__)

class ChatView(APIView):
    permission_classes = [AllowAny]
    """
    Handles both rendering the chat page (GET) and processing chat API requests (POST).
    Enhanced with blazing-fast turbo services, FAISS search, and performance monitoring.
    """
    def get(self, request, *args, **kwargs):
        """
        Renders the main chat interface.
        """
        return render(request, 'rag_agent/chat.html')

    def post(self, request, *args, **kwargs):
        """
        Handles chat messages with ultra-fast processing using turbo services.
        """
        start_time = time.time()
        rag_agent_config = apps.get_app_config('rag_agent')

        # Use turbo service if available, fallback to enhanced service
        reasoning_service = (
            rag_agent_config.turbo_reasoning_service or
            rag_agent_config.enhanced_reasoning_service
        )

        if not reasoning_service:
            return Response(
                {"error": "No reasoning service is available. Please check the server logs."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        question = request.data.get('question')
        session_id = request.data.get('session_id')  # Optional session ID for conversation context

        if not question:
            return Response(
                {"error": "A 'question' field is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Use the fastest available service
            service_type = "turbo" if rag_agent_config.turbo_reasoning_service else "enhanced"
            logger.info(f"Using {service_type} service for query with session_id: {session_id}")

            # Pass session_id for conversation context (always for turbo service)
            if service_type == "turbo":
                answer = reasoning_service.query(question, session_id=session_id)
            else:
                answer = reasoning_service.query(question)

            processing_time = time.time() - start_time

            # Log the interaction with timing
            logger.info(f"🚀 {service_type.title()} chat - {processing_time:.3f}s - Q: {question[:50]}... A: {answer[:50]}...")

            response_data = {
                "answer": answer,
                "processing_time": round(processing_time, 3),
                "service_type": service_type
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Chat processing failed after {processing_time:.3f}s: {e}", exc_info=True)
            return Response(
                {
                    "error": "An unexpected error occurred while processing your request.",
                    "processing_time": round(processing_time, 3)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PerformanceView(APIView):
    permission_classes = [AllowAny]
    """
    Provides performance metrics and system status.
    """
    def get(self, request, *args, **kwargs):
        """
        Returns performance metrics and system status.
        """
        try:
            rag_agent_config = apps.get_app_config('rag_agent')

            if rag_agent_config.turbo_reasoning_service:
                metrics = rag_agent_config.turbo_reasoning_service.get_performance_metrics()
                service_type = "turbo"
            else:
                metrics = {"message": "Turbo service not available"}
                service_type = "enhanced"

            response_data = {
                "service_type": service_type,
                "status": "operational",
                "metrics": metrics
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Performance metrics failed: {e}")
            return Response(
                {"error": "Failed to retrieve performance metrics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )