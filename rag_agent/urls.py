from django.urls import path
from .views import ChatView, PerformanceView

app_name = 'rag_agent'

urlpatterns = [
    path('', ChatView.as_view(), name='chat'),
    path('performance/', PerformanceView.as_view(), name='performance'),
]