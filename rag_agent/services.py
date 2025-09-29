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

# Prompt for generating the Cypher query
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher query to answer a question.
Instructions:
1. Use only the provided graph schema. Do not use any other node labels, relationship types, or properties that are not explicitly listed in the schema.
2. If the question cannot be answered using the provided schema, return the query: `RETURN "I am sorry, but I cannot answer this question based on the available information." AS result`
3. Return only the Cypher query, with no other text, explanation, or preamble.

Schema:
{schema}

Question: {question}
Cypher Query:
"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# Prompt for synthesizing the final answer after the query is executed
QA_TEMPLATE = """
You are a helpful AI assistant. Given the context below, answer the user's question.
The context is the result of a Cypher query. If the context is empty or contains the string "I am sorry...", it means the information was not found in the knowledge graph.
In that case, respond conversationally that you don't have the information. Do not mention the database or the query.

Context:
{context}

Question: {question}
Helpful Answer:
"""
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=QA_TEMPLATE
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
            qa_prompt=QA_PROMPT,
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