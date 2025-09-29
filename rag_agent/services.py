import logging
from django.conf import settings
from neo4j import GraphDatabase

# LangChain Imports
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document as LangchainDocument
from langchain.prompts.prompt import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom prompt template to guide the LLM in generating Cypher queries
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j developer who is an expert at writing Cypher queries.
Given the graph schema below, write a Cypher query that would answer the user's question.
Do not use any properties that are not in the schema. Do not use any relationship types that are not in the schema.
Return only the Cypher query, with no additional text or explanation.

Schema:
{schema}

Question: {question}
"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)


class KnowledgeGraphService:
    def __init__(self):
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_user = settings.NEO4J_USERNAME
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.neo4j_database = settings.NEO4J_DATABASE
        self.ollama_base_url = settings.OLLAMA_BASE_URL

        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database
        )

        self.llm = OllamaLLM(
            base_url=self.ollama_base_url,
            model="gemma3:1b",
            temperature=0
        )

        self.llm_transformer = LLMGraphTransformer(llm=self.llm)

    def text_to_graph(self, text: str):
        logger.info("Starting text-to-graph conversion...")
        documents = [LangchainDocument(page_content=text)]
        try:
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
            logger.info(f"Successfully converted text to {len(graph_documents)} graph documents.")
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
        logger.warning("Clearing the entire Neo4j database...")
        self.graph.query("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared successfully.")

class ReasoningService:
    def __init__(self):
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_user = settings.NEO4J_USERNAME
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.neo4j_database = settings.NEO4J_DATABASE
        self.ollama_base_url = settings.OLLAMA_BASE_URL

        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database
        )

        self.llm = OllamaLLM(
            base_url=self.ollama_base_url,
            model="gemma3:1b",
            temperature=0
        )

        self.graph.refresh_schema()

        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
        )

    def query(self, question: str) -> str:
        logger.info(f"Received query: {question}")
        try:
            result = self.cypher_chain.invoke({"query": question})
            logger.info(f"Successfully executed query. Result: {result}")
            return result.get('result', "I'm sorry, I couldn't find an answer based on the available information.")
        except Exception as e:
            logger.error(f"An error occurred during query execution: {e}")
            return "There was an error processing your request. Please check the logs."