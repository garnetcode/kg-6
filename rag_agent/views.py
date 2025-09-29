from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

def chat_view(request):
    """
    Renders the main chat interface.
    """
    return render(request, 'rag_agent/chat.html')

from django.apps import apps
from .services import ReasoningService

class ChatAPIView(APIView):
    """
    API endpoint for handling chat messages.
    This view receives a question from the user, passes it to the
    ReasoningService, and returns the KG-grounded answer.
    """
    def post(self, request, *args, **kwargs):
        # Access the shared ReasoningService instance from the app config.
        rag_agent_config = apps.get_app_config('rag_agent')
        reasoning_service = rag_agent_config.reasoning_service

        if not reasoning_service:
            return Response(
                {"error": "The Reasoning Service is not available. Please check the server logs."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        question = request.data.get('question')
        if not question:
            return Response(
                {"error": "A 'question' field is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            answer = reasoning_service.query(question)
            return Response({"answer": answer}, status=status.HTTP_200_OK)
        except Exception as e:
            # Log the exception details for debugging
            # logger.error(f"Error in ChatAPIView: {e}", exc_info=True)
            return Response(
                {"error": "An unexpected error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )