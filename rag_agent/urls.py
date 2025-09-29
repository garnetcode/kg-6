from django.urls import path
from .views import ChatView

app_name = 'rag_agent'

urlpatterns = [
    path('', ChatView.as_view(), name='chat'),
]