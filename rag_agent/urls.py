from django.urls import path
from . import views

app_name = 'rag_agent'

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('api/chat/', views.ChatAPIView.as_view(), name='chat_api'),
]