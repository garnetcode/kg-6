import logging
from django.conf import settings
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document as LangchainDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    def __init__(self):
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_user = settings.NEO4J_USERNAME
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.ollama_base_url = settings.OLLAMA_BASE_URL

        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password
        )

        self.llm = OllamaLLM(
            base_url=self.ollama_base_url,
            model="mistral",
            temperature=0
        )

        self.llm_transformer = LLMGraphTransformer(llm=self.llm)

    def text_to_graph(self, text: str):
        """
        Processes a text string, extracts graph data using an LLM,
        and adds it to the Neo4j database.
        """
        logger.info("Starting text-to-graph conversion...")

        # The LLMGraphTransformer expects a list of Langchain Documents
        documents = [LangchainDocument(page_content=text)]

        try:
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
            logger.info(f"Successfully converted text to {len(graph_documents)} graph documents.")

            # Add the extracted graph data to Neo4j
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            logger.info("Successfully added graph documents to Neo4j.")

        except Exception as e:
            logger.error(f"An error occurred during text-to-graph conversion: {e}")
            raise

    def clear_graph(self):
        """
        Deletes all nodes and relationships from the graph.
        """
        logger.warning("Clearing the entire Neo4j database...")
        self.graph.query("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared successfully.")

# Example of how to use this service
# from rag_agent.services import KnowledgeGraphService
# kg_service = KnowledgeGraphService()
# kg_service.text_to_graph("Elon Musk is the CEO of Tesla and SpaceX.")

from langchain.chains import GraphCypherQAChain

class ReasoningService:
    def __init__(self):
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_user = settings.NEO4J_USERNAME
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.ollama_base_url = settings.OLLAMA_BASE_URL

        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password
        )

        self.llm = OllamaLLM(
            base_url=self.ollama_base_url,
            model="mistral",
            temperature=0
        )

        # Refresh the graph schema to ensure the LLM has the latest context
        self.graph.refresh_schema()

        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            verbose=True, # Set to True for debugging to see generated Cypher
            allow_dangerous_requests=True,
        )

    def query(self, question: str) -> str:
        """
        Takes a user's question, generates a Cypher query, executes it,
        and returns a natural language answer.
        """
        logger.info(f"Received query: {question}")
        try:
            result = self.cypher_chain.invoke({"query": question})
            logger.info(f"Successfully executed query. Result: {result}")
            return result.get('result', "I'm sorry, I couldn't find an answer.")
        except Exception as e:
            logger.error(f"An error occurred during query execution: {e}")
            return "There was an error processing your request. Please check the logs."