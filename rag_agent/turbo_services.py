import logging
import json
import time
import asyncio
import hashlib
import uuid
from typing import List, Dict, Any, Tuple, Optional
from django.conf import settings
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime, timedelta

# High-performance imports
import faiss
from sentence_transformers import SentenceTransformer
import diskcache as dc
from joblib import Memory

# LangChain Imports
from langchain_neo4j.graphs import Neo4jGraph
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document as LangchainDocument
from langchain.prompts.prompt import PromptTemplate

logger = logging.getLogger(__name__)

# Import prompts from enhanced_services
from .enhanced_services import (
    ENTITY_EXTRACTION_TEMPLATE, SIMPLE_EXTRACTION_TEMPLATE,
    RELATIONSHIP_EXTRACTION_TEMPLATE
)

ENHANCED_CYPHER_TEMPLATE = """
Task: Generate a Cypher query to answer a question using the knowledge graph.

Instructions:
1. Use the provided graph schema and recent context.
2. If the question is conversational (greetings, general chat), return: `RETURN "CONVERSATIONAL_QUERY" AS result`
3. If asking about entities that might exist in the graph, try to find them with flexible matching.
4. For factual questions not in the graph, return: `RETURN "NO_GRAPH_DATA" AS result`
5. Use CONTAINS, STARTS WITH, or ENDS WITH for flexible text matching.
6. Return only the Cypher query, no explanations.

Schema:
{schema}

Recent Context (last few interactions):
{context}

Question: {query}
Cypher Query:
"""

ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"], template=ENTITY_EXTRACTION_TEMPLATE
)
SIMPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"], template=SIMPLE_EXTRACTION_TEMPLATE
)
RELATIONSHIP_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"], template=RELATIONSHIP_EXTRACTION_TEMPLATE
)
ENHANCED_CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "context", "query"], template=ENHANCED_CYPHER_TEMPLATE
)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def time_operation(self, operation_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start

                if operation_name not in self.metrics:
                    self.metrics[operation_name] = []
                self.metrics[operation_name].append(duration)

                logger.info(f"{operation_name}: {duration:.3f}s")
                return result
            return wrapper
        return decorator

    def get_average_time(self, operation_name):
        if operation_name in self.metrics:
            return np.mean(self.metrics[operation_name])
        return 0

# Global performance monitor
perf_monitor = PerformanceMonitor()

class ConversationGraph:
    """Manages conversation-level temporary graphs for context and coherence"""

    def __init__(self, session_id: str, driver, ttl_minutes: int = 60):
        self.session_id = session_id
        self.driver = driver
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.ttl_minutes = ttl_minutes
        self.conversation_entities = set()
        self.conversation_relationships = set()

        # Create conversation namespace in Neo4j
        self._initialize_conversation_namespace()
        logger.info(f"Initialized conversation graph for session {session_id}")

    def _initialize_conversation_namespace(self):
        """Create a conversation-specific namespace in Neo4j"""
        with self.driver.session() as session:
            # Create conversation node
            query = """
            MERGE (conv:Conversation {session_id: $session_id})
            SET conv.created_at = datetime(),
                conv.last_accessed = datetime()
            """
            session.run(query, session_id=self.session_id)

    def is_expired(self) -> bool:
        """Check if conversation has expired"""
        return datetime.now() - self.last_accessed > timedelta(minutes=self.ttl_minutes)

    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
        with self.driver.session() as session:
            query = """
            MATCH (conv:Conversation {session_id: $session_id})
            SET conv.last_accessed = datetime()
            """
            session.run(query, session_id=self.session_id)

    def add_conversation_entity(self, entity_name: str, entity_type: str, properties: Dict):
        """Add entity to conversation context"""
        self.touch()
        entity_key = f"{entity_type}:{entity_name}"
        self.conversation_entities.add(entity_key)

        with self.driver.session() as session:
            query = f"""
            MATCH (conv:Conversation {{session_id: $session_id}})
            MERGE (e:{entity_type} {{name: $entity_name}})
            SET e += $properties
            MERGE (conv)-[:MENTIONED]->(e)
            SET e.mentioned_in_session = $session_id,
                e.mentioned_at = datetime()
            """
            session.run(query,
                       session_id=self.session_id,
                       entity_name=entity_name,
                       properties=properties)

    def add_conversation_relationship(self, source: str, target: str, rel_type: str, properties: Dict):
        """Add relationship to conversation context"""
        self.touch()
        rel_key = f"{source}-{rel_type}-{target}"
        self.conversation_relationships.add(rel_key)

        with self.driver.session() as session:
            query = f"""
            MATCH (conv:Conversation {{session_id: $session_id}})
            MATCH (s), (t)
            WHERE s.name = $source_name AND t.name = $target_name
            MERGE (s)-[r:{rel_type}]->(t)
            SET r += $properties,
                r.conversation_id = $session_id,
                r.created_at = datetime()
            MERGE (conv)-[:HAS_RELATIONSHIP]->(r)
            """
            session.run(query,
                       session_id=self.session_id,
                       source_name=source,
                       target_name=target,
                       properties=properties)

    def add_conversation_message(self, question: str, answer: str):
        """Add conversation message (Q&A pair) to history"""
        self.touch()

        with self.driver.session() as session:
            query = """
            MATCH (conv:Conversation {session_id: $session_id})
            CREATE (msg:ConversationMessage {
                question: $question,
                answer: $answer,
                timestamp: datetime(),
                session_id: $session_id
            })
            CREATE (conv)-[:HAS_MESSAGE]->(msg)
            """
            session.run(query,
                       session_id=self.session_id,
                       question=question,
                       answer=answer)

    def get_conversation_context(self) -> str:
        """Get relevant context from conversation graph including message history"""
        self.touch()

        with self.driver.session() as session:
            # Get recent conversation messages
            message_query = """
            MATCH (conv:Conversation {session_id: $session_id})-[:HAS_MESSAGE]->(msg:ConversationMessage)
            RETURN msg.question as question, msg.answer as answer, msg.timestamp as timestamp
            ORDER BY msg.timestamp DESC
            LIMIT 5
            """

            # Get entities mentioned in this conversation
            entity_query = """
            MATCH (conv:Conversation {session_id: $session_id})-[:MENTIONED]->(e)
            RETURN e.name as name, labels(e)[0] as type, properties(e) as props
            ORDER BY e.mentioned_at DESC
            LIMIT 10
            """

            # Get relationships from this conversation
            rel_query = """
            MATCH (conv:Conversation {session_id: $session_id})-[:HAS_RELATIONSHIP]->(rel)
            MATCH (s)-[rel]->(t)
            RETURN s.name as source, type(rel) as rel_type, t.name as target, properties(rel) as props
            ORDER BY rel.created_at DESC
            LIMIT 10
            """

            # 🔍 CONTEXT RETRIEVAL DEBUGGING
            logger.info(f"🔍 Retrieving conversation context for session: {self.session_id}")

            try:
                messages = list(session.run(message_query, session_id=self.session_id))
                logger.info(f"📜 Retrieved {len(messages)} conversation messages")
            except Exception as e:
                logger.warning(f"⚠️ Failed to retrieve messages: {e}")
                messages = []

            try:
                entities = list(session.run(entity_query, session_id=self.session_id))
                logger.info(f"👤 Retrieved {len(entities)} entities")
            except Exception as e:
                logger.warning(f"⚠️ Failed to retrieve entities (expected if relationships missing): {e}")
                entities = []

            try:
                relationships = list(session.run(rel_query, session_id=self.session_id))
                logger.info(f"🔗 Retrieved {len(relationships)} relationships")
            except Exception as e:
                logger.warning(f"⚠️ Failed to retrieve relationships (expected if properties missing): {e}")
                relationships = []

            context_parts = []

            # Add recent conversation messages for immediate context
            if messages:
                context_parts.append("RECENT CONVERSATION HISTORY:")
                for msg in messages:
                    context_parts.append(f"User: {msg['question']}")
                    context_parts.append(f"Assistant: {msg['answer'][:100]}...")  # Truncate long answers
                    context_parts.append("---")

            # Add extracted entities
            if entities:
                context_parts.append("\nEXTRACTED ENTITIES:")
                for entity in entities:
                    context_parts.append(f"- {entity['name']} ({entity['type']})")

            # Add extracted relationships
            if relationships:
                context_parts.append("\nEXTRACTED RELATIONSHIPS:")
                for rel in relationships:
                    context_parts.append(f"- {rel['source']} {rel['rel_type']} {rel['target']}")

            # 🔍 CONTEXT BUILDING DEBUGGING
            logger.info(f"📜 Building context from {len(context_parts)} parts")

            if not context_parts:
                logger.warning("⚠️ No context parts found - returning empty context")
                return "No conversation context yet."

            final_context = "\n".join(context_parts)
            logger.info(f"📜 FINAL CONVERSATION CONTEXT ({len(final_context)} chars):")
            logger.info("="*40)
            logger.info(final_context)
            logger.info("="*40)

            return final_context

    def cleanup(self):
        """Clean up conversation graph data"""
        with self.driver.session() as session:
            # Remove conversation-specific data but keep main graph intact
            query = """
            MATCH (conv:Conversation {session_id: $session_id})
            OPTIONAL MATCH (conv)-[:MENTIONED]->(e)
            OPTIONAL MATCH (conv)-[:HAS_MESSAGE]->(msg)
            OPTIONAL MATCH (conv)-[:CONTAINS_RELATIONSHIP]->(r)
            DELETE conv, msg, r
            REMOVE e.mentioned_in_session, e.mentioned_at
            """
            session.run(query, session_id=self.session_id)
            logger.info(f"Cleaned up conversation graph for session {self.session_id}")

class ConversationManager:
    """Manages multiple conversation sessions"""

    def __init__(self, driver):
        self.driver = driver
        self.conversations = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.now()

    def get_or_create_conversation(self, session_id: str = None) -> ConversationGraph:
        """Get existing conversation or create new one"""
        # 🔍 SESSION MANAGEMENT DEBUGGING
        logger.info(f"🔄 ConversationManager.get_or_create_conversation called with session_id: {session_id}")

        if session_id is None:
            new_session_id = str(uuid.uuid4())
            logger.warning(f"⚠️ No session_id provided! Generated new UUID: {new_session_id}")
            session_id = new_session_id
        else:
            logger.info(f"✅ Using provided session_id: {session_id}")

        # Periodic cleanup
        if datetime.now() - self.last_cleanup > timedelta(seconds=self.cleanup_interval):
            self._cleanup_expired_conversations()

        if session_id not in self.conversations:
            logger.info(f"🆕 Creating new ConversationGraph for session: {session_id}")
            self.conversations[session_id] = ConversationGraph(session_id, self.driver)
        else:
            logger.info(f"♻️ Reusing existing ConversationGraph for session: {session_id}")

        # 🔍 VERIFY CONVERSATION OBJECT
        conversation = self.conversations[session_id]
        logger.info(f"🔍 ConversationGraph details: session_id={conversation.session_id}, created_at={conversation.created_at}")

        return conversation

    def _cleanup_expired_conversations(self):
        """Remove expired conversations"""
        expired_sessions = []
        for session_id, conv in self.conversations.items():
            if conv.is_expired():
                conv.cleanup()
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.conversations[session_id]

        self.last_cleanup = datetime.now()
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired conversations")

class FastEntityMatcher:
    """FAISS-powered ultra-fast entity similarity search"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info("Initializing FastEntityMatcher with FAISS...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = 384  # MiniLM embedding dimension

        # FAISS index for blazing fast similarity search
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.entity_names = []
        self.entity_data = {}

        # Cache for embeddings
        self.embedding_cache = dc.Cache('/tmp/claude/embeddings_cache', size_limit=100*1024*1024)  # 100MB

        logger.info("FastEntityMatcher initialized")

    @perf_monitor.time_operation("entity_embedding")
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get cached or compute embedding for text"""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        embedding = self.encoder.encode([text])[0]
        self.embedding_cache[cache_key] = embedding
        return embedding

    @perf_monitor.time_operation("entity_indexing")
    def add_entity(self, name: str, entity_type: str, properties: Dict):
        """Add entity to FAISS index"""
        embedding = self._get_embedding(name.lower())

        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)

        self.index.add(np.array([embedding], dtype=np.float32))
        self.entity_names.append(name)
        self.entity_data[name] = {"type": entity_type, "properties": properties}

    @perf_monitor.time_operation("similarity_search")
    def find_similar(self, query_name: str, threshold: float = 0.8, top_k: int = 5) -> List[Tuple[str, float]]:
        """Ultra-fast similarity search using FAISS"""
        if self.index.ntotal == 0:
            return []

        query_embedding = self._get_embedding(query_name.lower())
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # FAISS search - blazing fast!
        scores, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:
                results.append((self.entity_names[idx], float(score)))

        return results

    def bulk_index_entities(self, entities: List[Dict]):
        """Efficiently bulk index multiple entities"""
        if not entities:
            return

        names = [e["name"] for e in entities]
        embeddings = self.encoder.encode([name.lower() for name in names])

        # Normalize all embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Bulk add to FAISS
        self.index.add(embeddings.astype(np.float32))

        for entity in entities:
            name = entity["name"]
            self.entity_names.append(name)
            self.entity_data[name] = {
                "type": entity.get("type", "ENTITY"),
                "properties": entity.get("properties", {})
            }

class TurboEntityExtractor:
    """Ultra-fast async entity extraction with caching"""

    def __init__(self, ollama_base_url: str):
        self.llm = OllamaLLM(
            base_url=ollama_base_url,
            model="gemma3:1b",
            temperature=0.1
        )

        # Extraction cache
        self.extraction_cache = dc.Cache('/tmp/claude/extraction_cache', size_limit=50*1024*1024)  # 50MB

        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("TurboEntityExtractor initialized")

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    @perf_monitor.time_operation("cached_extraction")
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities with aggressive caching"""
        cache_key = self._cache_key(text)

        # Check cache first
        if cache_key in self.extraction_cache:
            logger.info("Cache hit for entity extraction")
            return self.extraction_cache[cache_key]

        # Extract using multi-attempt strategy
        result = self._extract_with_fallbacks(text)

        # Cache the result
        self.extraction_cache[cache_key] = result

        return result

    @perf_monitor.time_operation("llm_extraction")
    def _extract_with_fallbacks(self, text: str) -> Dict[str, List[Dict]]:
        """Multi-attempt extraction with performance optimization"""

        # Parallel attempt strategy - try multiple approaches concurrently
        futures = []

        # Submit extraction tasks to thread pool
        futures.append(
            self.executor.submit(self._attempt_extraction, ENTITY_EXTRACTION_PROMPT, text)
        )

        # Wait for first successful result with timeout
        for future in as_completed(futures, timeout=15):
            try:
                result = future.result()
                if self._is_valid_result(result):
                    logger.info("Fast extraction successful")
                    return result
            except Exception as e:
                logger.error(f"Extraction attempt failed: {e}")
                continue

        # Fallback to simplified extraction
        logger.info("Falling back to simplified extraction")
        return self._attempt_extraction(SIMPLE_EXTRACTION_PROMPT, text)

    def _attempt_extraction(self, prompt_template, text: str) -> Dict:
        """Single extraction attempt with robust JSON parsing"""
        try:
            prompt = prompt_template.format(text=text)
            response = self.llm.invoke(prompt)
            return self._robust_json_parse(response)
        except Exception as e:
            logger.error(f"Extraction attempt failed: {e}")
            return {"entities": [], "relationships": []}

    def _robust_json_parse(self, response: str) -> Dict:
        """Optimized JSON parsing"""
        response = response.strip()

        # Fast JSON parsing attempts
        parsers = [
            lambda r: json.loads(r),
            lambda r: json.loads(self._extract_json_block(r)),
            lambda r: json.loads(self._find_json_structure(r)),
        ]

        for parser in parsers:
            try:
                return parser(response)
            except:
                continue

        # Fallback to pattern extraction
        return self._extract_partial_json(response)

    def _extract_json_block(self, response: str) -> str:
        """Extract JSON from code blocks"""
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        return match.group(1) if match else response

    def _find_json_structure(self, response: str) -> str:
        """Find JSON-like structure"""
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return match.group(0) if match else '{}'

    def _extract_partial_json(self, response: str) -> Dict:
        """Fast pattern-based extraction"""
        import re

        entities = []
        relationships = []

        # Quick entity extraction
        entity_matches = re.findall(r'"name":\s*"([^"]+)".*?"type":\s*"([^"]+)"', response, re.IGNORECASE)
        for name, entity_type in entity_matches:
            entities.append({"name": name, "type": entity_type, "properties": {}})

        # Quick relationship extraction
        rel_matches = re.findall(r'"source":\s*"([^"]+)".*?"target":\s*"([^"]+)".*?"type":\s*"([^"]+)"', response, re.IGNORECASE)
        for source, target, rel_type in rel_matches:
            relationships.append({
                "source": source, "target": target, "type": rel_type, "properties": {}
            })

        return {"entities": entities, "relationships": relationships}

    def _is_valid_result(self, result: Dict) -> bool:
        """Quick validation"""
        return (
            result and isinstance(result, dict) and
            "entities" in result and "relationships" in result and
            len(result.get("relationships", [])) > 0
        )

class TurboKnowledgeGraphService:
    """Ultra-fast KG service with FAISS and optimized Neo4j operations"""

    def __init__(self):
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_user = settings.NEO4J_USERNAME
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.neo4j_database = settings.NEO4J_DATABASE
        self.ollama_base_url = settings.OLLAMA_BASE_URL

        # Connection pool for Neo4j
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
            max_connection_pool_size=50,
            connection_acquisition_timeout=30
        )

        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database
        )

        # Ultra-fast components
        self.entity_matcher = FastEntityMatcher()
        self.entity_extractor = TurboEntityExtractor(self.ollama_base_url)

        # Initialize entity index
        self._initialize_entity_index()

        logger.info("TurboKnowledgeGraphService initialized")

    @perf_monitor.time_operation("entity_index_init")
    def _initialize_entity_index(self):
        """Load existing entities into FAISS index"""
        try:
            # Batch load existing entities
            query = """
            MATCH (n)
            RETURN n.name as name, labels(n) as types, properties(n) as props
            LIMIT 1000
            """

            entities = self.graph.query(query)
            entity_data = []

            for entity in entities:
                if entity["name"]:
                    entity_data.append({
                        "name": entity["name"],
                        "type": entity["types"][0] if entity["types"] else "ENTITY",
                        "properties": entity["props"]
                    })

            # Bulk index entities
            if entity_data:
                self.entity_matcher.bulk_index_entities(entity_data)
                logger.info(f"Indexed {len(entity_data)} entities in FAISS")

        except Exception as e:
            logger.error(f"Failed to initialize entity index: {e}")

    @perf_monitor.time_operation("fast_entity_storage")
    def extract_and_store_entities(self, text: str, source: str = "conversation") -> Dict:
        """Ultra-fast entity extraction and storage"""
        start_time = time.time()

        extraction_result = self.entity_extractor.extract_entities(text)

        entities_added = 0
        relationships_added = 0
        entity_name_mapping = {}

        try:
            # Batch process entities
            entity_batch = []
            for entity in extraction_result.get("entities", []):
                entity_name = entity.get("name", "").strip()
                entity_type = entity.get("type", "ENTITY")
                properties = entity.get("properties", {})
                properties["source"] = source

                if entity_name:
                    # Fast similarity search
                    similar = self.entity_matcher.find_similar(entity_name, threshold=0.85)

                    if similar:
                        # Use existing entity
                        standardized_name = similar[0][0]
                        entity_name_mapping[entity_name] = standardized_name
                        logger.info(f"Matched entity: {entity_name} -> {standardized_name}")
                    else:
                        # Create new entity
                        entity_batch.append({
                            "name": entity_name,
                            "type": entity_type,
                            "properties": properties
                        })
                        entity_name_mapping[entity_name] = entity_name

                        # Add to FAISS index
                        self.entity_matcher.add_entity(entity_name, entity_type, properties)

                    entities_added += 1

            # Batch create entities in Neo4j
            if entity_batch:
                self._batch_create_entities(entity_batch)

            # Batch create relationships
            relationship_batch = []
            for rel in extraction_result.get("relationships", []):
                source_name = rel.get("source", "").strip()
                target_name = rel.get("target", "").strip()
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {})
                properties["extracted_from"] = source

                if source_name and target_name:
                    final_source = entity_name_mapping.get(source_name, source_name)
                    final_target = entity_name_mapping.get(target_name, target_name)

                    relationship_batch.append({
                        "source": final_source,
                        "target": final_target,
                        "type": rel_type,
                        "properties": properties
                    })
                    relationships_added += 1

            # Batch create relationships
            if relationship_batch:
                self._batch_create_relationships(relationship_batch)

        except Exception as e:
            logger.error(f"Error in fast entity storage: {e}", exc_info=True)

        total_time = time.time() - start_time
        logger.info(f"Fast extraction completed in {total_time:.3f}s: {entities_added} entities, {relationships_added} relationships")

        return {
            "entities_added": entities_added,
            "relationships_added": relationships_added,
            "extraction_result": extraction_result,
            "entity_mapping": entity_name_mapping,
            "processing_time": total_time
        }

    @perf_monitor.time_operation("batch_entity_creation")
    def _batch_create_entities(self, entities: List[Dict]):
        """Optimized batch entity creation with error handling"""
        with self.driver.session() as session:
            def create_entities_tx(tx, entities):
                for entity in entities:
                    # Sanitize entity type to be valid Cypher label
                    entity_type = entity['type'].replace(' ', '_').replace('-', '_')
                    query = f"""
                    MERGE (e:{entity_type} {{name: $name}})
                    SET e += $properties
                    """
                    try:
                        result = tx.run(query, name=entity["name"], properties=entity["properties"])
                        result.consume()
                    except Exception as e:
                        logger.warning(f"Failed to create entity {entity['name']}: {e}")

            session.execute_write(create_entities_tx, entities)

    @perf_monitor.time_operation("batch_relationship_creation")
    def _batch_create_relationships(self, relationships: List[Dict]):
        """Optimized batch relationship creation with improved query"""
        with self.driver.session() as session:
            def create_relationships_tx(tx, relationships):
                for rel in relationships:
                    # Create relationship between any nodes with matching names
                    query = f"""
                    MATCH (s {{name: $source_name}})
                    MATCH (t {{name: $target_name}})
                    WITH s, t
                    LIMIT 1
                    MERGE (s)-[r:{rel['type']}]->(t)
                    SET r += $properties
                    """
                    try:
                        result = tx.run(query,
                              source_name=rel["source"],
                              target_name=rel["target"],
                              properties=rel["properties"])
                        # Consume result to ensure query completes
                        result.consume()
                    except Exception as e:
                        logger.warning(f"Failed to create relationship {rel['source']} -> {rel['target']}: {e}")

            session.execute_write(create_relationships_tx, relationships)

class TurboReasoningService:
    """Ultra-fast reasoning service with performance optimization and conversation context"""

    def __init__(self):
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_user = settings.NEO4J_USERNAME
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.neo4j_database = settings.NEO4J_DATABASE
        self.ollama_base_url = settings.OLLAMA_BASE_URL

        # Initialize Neo4j driver for conversation management
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )

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

        self.kg_service = TurboKnowledgeGraphService()

        # Conversation management
        self.conversation_manager = ConversationManager(self.driver)

        # Query cache
        self.query_cache = dc.Cache('/tmp/claude/query_cache', size_limit=25*1024*1024)  # 25MB

        # Refresh schema and initialize chain
        self.graph.refresh_schema()

        # Enhanced QA prompt with strong conversation memory focus
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""
You are a helpful neuro-symbolic AI assistant with perfect conversation memory.

CONVERSATION CONTEXT:
{context}

CURRENT QUESTION: {query}

CRITICAL INSTRUCTIONS:
1. **PRIORITIZE CONVERSATION CONTEXT**: Always check the conversation history FIRST before responding
2. **PERFECT MEMORY**: If someone mentioned their name, company, or other details in our conversation, I MUST remember them exactly
3. **DIRECT QUESTIONS**: For questions like "What is my name?" or "Where do I work?", use the conversation history to answer accurately
4. **CONVERSATIONAL CONTINUITY**: Reference previous topics naturally to maintain coherent conversation flow
5. **CONTEXT OVER GENERAL KNOWLEDGE**: If there's a conflict between conversation context and general knowledge, ALWAYS prioritize what was said in our conversation

EXAMPLES:
- If conversation shows "Hi my name is Alice", then "What is my name?" → "Your name is Alice"
- If conversation shows "I work at DataCorp", then "Where do I work?" → "You work at DataCorp"

Helpful Answer based on our conversation:
"""
        )

        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=ENHANCED_CYPHER_PROMPT,
            qa_prompt=self.qa_prompt,
        )

        logger.info("TurboReasoningService initialized")

    @perf_monitor.time_operation("total_query_time")
    def query(self, question: str, session_id: str = None) -> str:
        """Ultra-fast query processing with reliable conversation memory"""
        logger.info(f"🔍 Processing turbo query: {question} (session_id: {session_id})")

        # SESSION VALIDATION AND DEBUGGING
        if session_id is None:
            logger.warning("⚠️ No session_id provided! This will create a new random session.")
        else:
            logger.info(f"✅ Received session_id: {session_id}")

        # Get or create conversation context
        conversation = self.conversation_manager.get_or_create_conversation(session_id)
        logger.info(f"🗣️ Using conversation session: {conversation.session_id}")

        # CRITICAL: Verify session ID consistency
        if session_id and conversation.session_id != session_id:
            logger.error(f"❌ SESSION ID MISMATCH! Provided: {session_id}, Got: {conversation.session_id}")
        else:
            logger.info(f"✅ Session ID consistent: {conversation.session_id}")

        # Disable caching for conversation-dependent queries to prevent contamination
        # TODO: Implement smarter session-aware caching later

        try:
            start_time = time.time()

            # STEP 1: Store current question immediately for conversation continuity
            logger.info("Storing current question for conversation history...")
            try:
                # Store question immediately (we'll update with answer later)
                conversation.add_conversation_message(question, "[Processing...]")
            except Exception as e:
                logger.warning(f"Failed to store question: {e}")

            # STEP 2: Simple entity extraction with fallback
            logger.info("Performing simple entity extraction...")
            entities, relationships = self._simple_entity_extraction(question)

            # STEP 3: Get conversation context (now includes current question)
            conversation_context = conversation.get_conversation_context()
            full_context = f"{conversation_context}\n\nCurrent Query: {question}"

            # 🔍 REAL-TIME CONTEXT DEBUGGING
            logger.info(f"📜 Context retrieved ({len(full_context)} chars)")
            logger.info("📜 FULL CONVERSATION CONTEXT:")
            logger.info("="*50)
            logger.info(f"{conversation_context}")
            logger.info("="*50)
            logger.info(f"📝 Current Query: {question}")

            # STEP 4: Generate LLM response with full context
            logger.info("🤖 Sending request to LLM with full context...")

            # 🔍 REAL-TIME LLM PROMPT DEBUGGING
            llm_prompt_data = {
                "query": question,
                "context": full_context
            }
            logger.info("🤖 LLM PROMPT DATA:")
            logger.info(f"   Query: {question}")

            # 🎯 CRITICAL FIX: For conversational questions, use direct QA instead of Cypher generation
            # IMPROVED: Distinguish between QUESTIONS (for memory recall) and STATEMENTS (for learning)
            conversational_question_patterns = [
                "what is my name", "what's my name", "who am i",
                "where do i work", "what company do i work", "where am i employed",
                "what did i say", "what did we discuss", "what did i tell you",
                "do you remember", "what do you know about me",
                "who do i work for", "what is my job", "where is my workplace"
            ]

            # More precise detection: must be a question (contains question words + question structure)
            question_lower = question.lower().strip()

            # Check for question patterns AND question structure (starts with question word OR ends with ?)
            has_question_word = any(question_lower.startswith(word) for word in ['what', 'who', 'where', 'when', 'why', 'how', 'do', 'can', 'will', 'would', 'could', 'should'])
            has_question_mark = question_lower.endswith('?')
            matches_conversational_pattern = any(pattern in question_lower for pattern in conversational_question_patterns)

            # CRITICAL: Only trigger for actual questions, not statements
            is_memory_question = matches_conversational_pattern and (has_question_word or has_question_mark)

            # Broader check for general conversational queries (e.g., greetings)
            is_greeting = question_lower.strip() in ["hi", "hello", "hey", "greetings"]
            is_conversational = is_memory_question or is_greeting

            # Enhanced debugging
            logger.info(f"🔍 Question Analysis: '{question_lower}'")
            logger.info(f"   Has question word: {has_question_word}")
            logger.info(f"   Has question mark: {has_question_mark}")
            logger.info(f"   Matches pattern: {matches_conversational_pattern}")
            logger.info(f"   Is memory question: {is_memory_question}")
            logger.info(f"   Is greeting: {is_greeting}")
            logger.info(f"   Is conversational: {is_conversational}")

            if is_greeting or (is_memory_question and len(full_context) > 50):
                logger.info("🎯 Detected conversational question - using direct QA approach")
                # Use QA prompt directly with conversation context
                qa_input = self.qa_prompt.format(context=full_context, query=question)
                logger.info("🤖 DIRECT QA PROMPT:")
                logger.info("="*50)
                logger.info(qa_input[:500] + "..." if len(qa_input) > 500 else qa_input)
                logger.info("="*50)

                raw_answer = self.llm.invoke(qa_input)
                logger.info("🎯 Using direct QA response")
            else:
                logger.info("🔍 Using GraphCypherQAChain for database query")
                result = self.cypher_chain.invoke(llm_prompt_data)
                raw_answer = result.get('result', 'I apologize, but I encountered an issue processing your request.')

            # 🔍 REAL-TIME LLM RESPONSE DEBUGGING
            logger.info("🤖 RAW LLM RESPONSE:")
            logger.info("="*50)
            logger.info(f"{raw_answer}")
            logger.info("="*50)

            answer = raw_answer

            # STEP 5: Update conversation with final answer and store entities
            try:
                # Update the conversation message with the real answer
                conversation.add_conversation_message(question, answer)

                # Store extracted entities
                for entity in entities:
                    conversation.add_conversation_entity(
                        entity['name'], entity['type'], entity.get('properties', {})
                    )

                for rel in relationships:
                    conversation.add_conversation_relationship(
                        rel['source'], rel['target'], rel['type'], rel.get('properties', {})
                    )

                logger.info(f"Updated conversation with answer and stored {len(entities)} entities, {len(relationships)} relationships")
            except Exception as e:
                logger.warning(f"Failed to update conversation data: {e}")

            total_time = time.time() - start_time
            logger.info(f"Turbo query completed in {total_time:.3f}s")
            return answer

        except Exception as e:
            logger.error(f"Turbo query failed: {e}")
            return "I apologize, but I encountered an issue processing your request. Please try again."

    def _simple_entity_extraction(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Simple entity extraction with regex fallbacks for reliability"""
        import re

        entities = []
        relationships = []

        try:
            # Try LLM extraction first (with shorter timeout)
            extraction_result = self.kg_service.extract_and_store_entities(text, "user_question")
            if extraction_result:
                entities = extraction_result.get('entities', [])
                relationships = extraction_result.get('relationships', [])
                logger.info(f"LLM extracted: {len(entities)} entities, {len(relationships)} relationships")
                return entities, relationships
        except Exception as e:
            logger.warning(f"LLM extraction failed, using fallback: {e}")

        # 🔧 ENHANCED: Improved regex patterns for more robust entity extraction
        logger.info(f"🔧 Starting enhanced regex extraction for: '{text}'")

        # Extract names - improved patterns with better boundary detection
        name_patterns = [
            r"(?:my name is|i'm|i am)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",  # Multi-word names
            r"(?:hi|hello),?\s*(?:my name is|i'm|i am)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"(?:called|named)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"i'm\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*?)(?:\s+(?:and|,|who|from)|\s*$)"
        ]

        extracted_names = []
        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if 1 < len(name) < 50 and name not in extracted_names:  # Avoid duplicates
                    extracted_names.append(name)
                    entities.append({
                        'name': name,
                        'type': 'Person',
                        'properties': {'source': 'regex_fallback', 'extraction_method': 'name_pattern'}
                    })
                    logger.info(f"🔧 Regex extracted name: '{name}' from pattern: {pattern}")

        # Extract companies - improved patterns
        company_patterns = [
            r"(?:work at|work for|employed by|employee at)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"(?:company|employer|organization)(?:\s+is)?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"(?:i|I)\s+work\s+(?:at|for)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)"
        ]

        extracted_companies = []
        for pattern in company_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company = match.group(1).strip()
                if 1 < len(company) < 50 and company not in extracted_companies:  # Avoid duplicates
                    extracted_companies.append(company)
                    entities.append({
                        'name': company,
                        'type': 'Organization',
                        'properties': {'source': 'regex_fallback', 'extraction_method': 'company_pattern'}
                    })
                    logger.info(f"🔧 Regex extracted company: '{company}' from pattern: {pattern}")

        # 🔗 ENHANCED: Create relationships between extracted entities
        for name in extracted_names:
            for company in extracted_companies:
                relationships.append({
                    'source': name,
                    'target': company,
                    'type': 'WORKS_FOR',
                    'properties': {
                        'source': 'regex_fallback',
                        'extraction_method': 'name_company_relationship',
                        'confidence': 0.8
                    }
                })
                logger.info(f"🔗 Created relationship: {name} WORKS_FOR {company}")

        logger.info(f"🔧 Enhanced regex extraction completed: {len(entities)} entities, {len(relationships)} relationships")
        return entities, relationships

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            "extraction_avg": perf_monitor.get_average_time("llm_extraction"),
            "similarity_search_avg": perf_monitor.get_average_time("similarity_search"),
            "storage_avg": perf_monitor.get_average_time("fast_entity_storage"),
            "query_avg": perf_monitor.get_average_time("total_query_time"),
            "cache_sizes": {
                "extraction_cache": len(self.kg_service.entity_extractor.extraction_cache),
                "embedding_cache": len(self.kg_service.entity_matcher.embedding_cache),
                "query_cache": len(self.query_cache)
            },
            "active_conversations": len(self.conversation_manager.conversations)
        }

    def debug_conversation(self, session_id: str) -> Dict:
        """Debug conversation state for troubleshooting"""
        try:
            conversation = self.conversation_manager.get_or_create_conversation(session_id)
            context = conversation.get_conversation_context()

            return {
                "session_id": session_id,
                "conversation_session_id": conversation.session_id,
                "context_length": len(context),
                "context_preview": context[:500] + "..." if len(context) > 500 else context,
                "created_at": conversation.created_at.isoformat(),
                "last_accessed": conversation.last_accessed.isoformat(),
                "entities_count": len(conversation.conversation_entities),
                "relationships_count": len(conversation.conversation_relationships)
            }
        except Exception as e:
            return {"error": str(e), "session_id": session_id}