from django.db import models

class Document(models.Model):
    """
    Represents a source document that has been ingested into the system.
    """
    title = models.CharField(max_length=255)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.title

class ConversationHistory(models.Model):
    """
    Stores the history of a conversation between a user and the agent.
    """
    session_id = models.CharField(max_length=100, unique=True)
    history = models.JSONField(default=list) # Stores a list of {"user": "...", "assistant": "..."}

    def __str__(self):
        return f"Conversation {self.session_id}"