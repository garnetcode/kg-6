from django.core.management.base import BaseCommand
from langchain_neo4j import Neo4jGraph
from django.conf import settings

class Command(BaseCommand):
    help = 'Checks the status of the Neo4j graph by counting the number of nodes.'

    def handle(self, *args, **options):
        self.stdout.write("Connecting to Neo4j to check graph status...")

        try:
            graph = Neo4jGraph(
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD,
                database=settings.NEO4J_DATABASE
            )

            # Query to count all nodes
            result = graph.query("MATCH (n) RETURN count(n) AS node_count")

            if result and len(result) > 0 and "node_count" in result[0]:
                node_count = result[0]["node_count"]
                if node_count > 0:
                    self.stdout.write(self.style.SUCCESS(f"Success! The graph contains {node_count} nodes."))
                else:
                    self.stdout.write(self.style.WARNING("The graph is empty. No nodes were found."))
                    self.stdout.write(self.style.NOTICE("Please run the `ingest_data` command to populate the graph."))
            else:
                self.stderr.write(self.style.ERROR("Could not retrieve node count from the graph. The query returned an unexpected result."))

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"An error occurred while connecting to or querying Neo4j: {e}"))
            self.stderr.write(self.style.ERROR("Please ensure Neo4j is running and the connection details in your .env file are correct."))