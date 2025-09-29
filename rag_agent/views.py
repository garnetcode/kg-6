from django.shortcuts import render
from django.apps import apps
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class ChatView(APIView):
    """
    Handles both rendering the chat page (GET) and processing chat API requests (POST).
    """
    def get(self, request, *args, **kwargs):
        """
        Renders the main chat interface.
        """
        return render(request, 'rag_agent/chat.html')

    def post(self, request, *args, **kwargs):
        """
        Handles chat messages from the user, passes them to the
        ReasoningService, and returns the KG-grounded answer.
        """
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
            return Response(
                {"error": "An unexpected error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )