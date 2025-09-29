import logging
from django.core.management.base import BaseCommand
from django.utils import timezone
from rag_agent.models import Document
from rag_agent.services import KnowledgeGraphService

logger = logging.getLogger(__name__)

from django.conf import settings

class Command(BaseCommand):
    help = 'Ingests a source document into the Knowledge Graph.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear the existing graph before ingesting.',
        )
        parser.add_argument(
            '--file-path',
            type=str,
            default='source_document.txt',
            help='Path to the source document to ingest.',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting data ingestion process..."))

        kg_service = KnowledgeGraphService()
        file_path = options['file_path']

        if options['clear']:
            self.stdout.write(self.style.WARNING("Clearing existing data from the knowledge graph."))
            kg_service.clear_graph()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_text = f.read()
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"Source document not found at: {file_path}"))
            return

        # Create or retrieve the document record in Django's DB
        doc, created = Document.objects.get_or_create(
            title=file_path,
            defaults={'content': source_text}
        )

        if not created and doc.processed_at:
            self.stdout.write(self.style.NOTICE(f"Document '{file_path}' has already been processed. Use --clear to re-ingest."))
            if not options['clear']:
                return

        self.stdout.write(f"Processing document: '{doc.title}'")

        try:
            # The core step: convert text to graph and load into Neo4j
            kg_service.text_to_graph(doc.content)

            # Mark the document as processed
            doc.processed_at = timezone.now()
            doc.save()

            self.stdout.write(self.style.SUCCESS(f"Successfully processed and ingested document '{doc.title}'."))

        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            self.stderr.write(self.style.ERROR(f"An error occurred: {e}"))
            self.stderr.write(self.style.ERROR(
                "Please ensure Ollama and Neo4j are running and accessible. "
                "You can start them with 'docker-compose up -d'."
            ))