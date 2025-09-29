import logging
import json
from typing import List, Dict, Any, Tuple
from django.conf import settings
from neo4j import GraphDatabase

# LangChain Imports
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document as LangchainDocument
from langchain.prompts.prompt import PromptTemplate

logger = logging.getLogger(__name__)

# Open-domain business relationship extraction prompt
ENTITY_EXTRACTION_TEMPLATE = """
TASK: Extract entities and relationships from business/conversational text. Discover ANY type of relationship based on context!

APPROACH:
1. Identify ALL entities (people, places, organizations, documents, projects, equipment, etc.)
2. Determine entity types from context (don't limit to predefined types)
3. Extract relationships by analyzing verbs and context - ANY relationship that connects entities
4. Infer relationship types from the action/verb used

BUSINESS EXAMPLES:
- "John issued fuel coupon #123" → John ISSUED fuel_coupon_123
- "Sarah approved the marketing budget" → Sarah APPROVED marketing_budget
- "I work for Bowspace" → [Speaker] WORKS_FOR Bowspace
- "Team Alpha assigned to Project Phoenix" → Team_Alpha ASSIGNED_TO Project_Phoenix
- "Invoice INV-001 paid by ABC Corp" → INV_001 PAID_BY ABC_Corp
- "Mike reports to Jennifer" → Mike REPORTS_TO Jennifer
- "Alice manages the sales team" → Alice MANAGES sales_team

PERSONAL EXAMPLES:
- "I live in New York" → [Speaker] LIVES_IN New_York
- "John knows Sarah" → John KNOWS Sarah
- "I love pizza" → [Speaker] LIKES pizza

TEXT: {text}

INSTRUCTIONS:
1. Extract ALL entities with appropriate types based on context
2. For each verb/action, create a relationship using the verb as the relationship type
3. Use [Speaker] or [User] for first-person references (I, my, me)
4. Relationship types should be UPPERCASE verbs (WORKS_FOR, ISSUED, APPROVED, MANAGES, etc.)
5. Entity types should reflect their nature (PERSON, COMPANY, DOCUMENT, PROJECT, LOCATION, CONCEPT, etc.)

RETURN VALID JSON:
{{
  "entities": [
    {{"type": "PERSON", "name": "John", "properties": {{}}}},
    {{"type": "COMPANY", "name": "Bowspace", "properties": {{}}}},
    {{"type": "DOCUMENT", "name": "fuel_coupon_123", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "John", "target": "Bowspace", "type": "WORKS_FOR", "properties": {{}}}},
    {{"source": "John", "target": "fuel_coupon_123", "type": "ISSUED", "properties": {{}}}}
  ]
}}
"""

ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"], template=ENTITY_EXTRACTION_TEMPLATE
)

# Simplified fallback prompt for when primary extraction fails
SIMPLE_EXTRACTION_TEMPLATE = """
Extract entities and relationships from this text. Focus on finding WHO does WHAT to WHOM.

Text: {text}

Find:
1. Names of people, companies, places, documents, projects
2. Actions/verbs that connect them (works for, issued, approved, manages, etc.)

Return simple JSON with entities and relationships:
{{
  "entities": [{{"name": "EntityName", "type": "ENTITY_TYPE"}}],
  "relationships": [{{"source": "Entity1", "target": "Entity2", "type": "ACTION_VERB"}}]
}}
"""

SIMPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"], template=SIMPLE_EXTRACTION_TEMPLATE
)

# Relationship-focused prompt for final attempt
RELATIONSHIP_EXTRACTION_TEMPLATE = """
Focus ONLY on finding relationships in this text: {text}

What actions connect the entities? Look for verbs like: works, issued, approved, manages, reports, assigned, paid, etc.

Return ONLY relationships as JSON:
{{
  "relationships": [{{"source": "Who", "target": "What/Whom", "type": "ACTION"}}]
}}
"""

RELATIONSHIP_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"], template=RELATIONSHIP_EXTRACTION_TEMPLATE
)

# Enhanced Cypher generation for context-aware queries
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

Question: {question}
Cypher Query:
"""

ENHANCED_CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "context", "question"], template=ENHANCED_CYPHER_TEMPLATE
)

class EntityExtractor:
    """Extracts entities from text using LLM-based NER"""

    def __init__(self, ollama_base_url: str):
        self.llm = OllamaLLM(
            base_url=ollama_base_url,
            model="gemma3:1b",
            temperature=0.1
        )

    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities and relationships using multi-attempt LLM strategy"""

        # Attempt 1: Full business-context prompt
        logger.info("Attempt 1: Full business-context extraction")
        result = self._attempt_extraction(ENTITY_EXTRACTION_PROMPT, text)
        if self._is_valid_result(result):
            logger.info("Success with primary extraction")
            return result

        # Attempt 2: Simplified prompt
        logger.info("Attempt 2: Simplified extraction")
        result = self._attempt_extraction(SIMPLE_EXTRACTION_PROMPT, text)
        if self._is_valid_result(result):
            logger.info("Success with simplified extraction")
            return result

        # Attempt 3: Relationship-focused prompt + basic entity extraction
        logger.info("Attempt 3: Relationship-focused extraction")
        rel_result = self._attempt_extraction(RELATIONSHIP_EXTRACTION_PROMPT, text)
        entities = self._extract_basic_entities(text)

        if rel_result and "relationships" in rel_result:
            logger.info("Success with relationship-focused extraction")
            return {
                "entities": entities,
                "relationships": rel_result["relationships"]
            }

        # Final fallback: Empty result
        logger.warning("All LLM extraction attempts failed")
        return {"entities": [], "relationships": []}

    def _attempt_extraction(self, prompt_template, text: str) -> Dict:
        """Single extraction attempt with robust JSON parsing"""
        try:
            prompt = prompt_template.format(text=text)
            response = self.llm.invoke(prompt)

            # Try multiple JSON parsing strategies
            return self._robust_json_parse(response)

        except Exception as e:
            logger.error(f"Extraction attempt failed: {e}")
            return {}

    def _robust_json_parse(self, response: str) -> Dict:
        """Robust JSON parsing with multiple fallback strategies"""
        response = response.strip()

        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON-like structure
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Partial JSON parsing (extract what we can)
        return self._extract_partial_json(response)

    def _extract_partial_json(self, response: str) -> Dict:
        """Extract entities and relationships even from malformed JSON"""
        import re

        entities = []
        relationships = []

        # Extract entity-like patterns
        entity_patterns = [
            r'"name":\s*"([^"]+)".*?"type":\s*"([^"]+)"',
            r'"([^"]+)".*?(?:PERSON|COMPANY|ORGANIZATION|LOCATION|DOCUMENT|PROJECT)'
        ]

        for pattern in entity_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    entities.append({"name": match[0], "type": match[1], "properties": {}})

        # Extract relationship-like patterns
        rel_patterns = [
            r'"source":\s*"([^"]+)".*?"target":\s*"([^"]+)".*?"type":\s*"([^"]+)"',
            r'([^"]+)\s+(?:WORKS_FOR|ISSUED|APPROVED|MANAGES|REPORTS_TO)\s+([^"]+)'
        ]

        for pattern in rel_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 3:
                    relationships.append({
                        "source": match[0],
                        "target": match[1],
                        "type": match[2],
                        "properties": {}
                    })

        return {"entities": entities, "relationships": relationships}

    def _is_valid_result(self, result: Dict) -> bool:
        """Check if extraction result is valid and contains relationships"""
        return (
            result and
            isinstance(result, dict) and
            "entities" in result and
            "relationships" in result and
            len(result.get("relationships", [])) > 0
        )

    def _extract_basic_entities(self, text: str) -> List[Dict]:
        """Extract basic entities using simple patterns"""
        import re

        entities = []
        words = text.split()

        # Look for capitalized words as potential entities
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                entities.append({
                    "name": clean_word,
                    "type": "ENTITY",
                    "properties": {}
                })

        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity["name"] not in seen:
                unique_entities.append(entity)
                seen.add(entity["name"])

        return unique_entities

    def _enhanced_fallback_extraction(self, text: str) -> Dict[str, List[Dict]]:
        """Enhanced fallback method with pattern-based relationship detection"""
        import re

        entities = []
        relationships = []

        # Enhanced entity extraction
        words = text.split()

        # Look for capitalized words as potential entities
        for word in words:
            if word[0].isupper() and len(word) > 2:
                clean_word = word.strip(".,!?;:")
                entities.append({
                    "type": "ENTITY",
                    "name": clean_word,
                    "properties": {}
                })

        # Pattern-based relationship extraction
        text_lower = text.lower()

        # Extract "I am X from Y" pattern
        from_pattern = re.findall(r'i am (\w+) from (\w+)', text_lower)
        for person, place in from_pattern:
            entities.extend([
                {"type": "PERSON", "name": person.title(), "properties": {}},
                {"type": "PLACE", "name": place.title(), "properties": {}}
            ])
            relationships.append({
                "source": person.title(),
                "target": place.title(),
                "type": "LIVES_IN",
                "properties": {}
            })

        # Enhanced work relationship patterns
        work_patterns = [
            (r'(\w+) works? at (\w+)', "WORKS_AT"),
            (r'(\w+) works? for (\w+)', "WORKS_FOR"),
            (r'(\w+) employed by (\w+)', "EMPLOYED_BY"),
            (r'(\w+) job at (\w+)', "WORKS_AT"),
            (r'(\w+) works? with (\w+)', "COLLABORATES_WITH"),
            (r'i work for (\w+)', "WORKS_FOR"),  # Handle first person
            (r'i work at (\w+)', "WORKS_AT"),
            (r'my job is at (\w+)', "WORKS_AT"),
        ]

        for pattern, relationship_type in work_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    person, org = match
                    person = person.title() if person != "i" else "User"
                else:
                    person = "User"  # First person patterns
                    org = match

                entities.extend([
                    {"type": "PERSON", "name": person, "properties": {}},
                    {"type": "ORGANIZATION", "name": org.title(), "properties": {}}
                ])
                relationships.append({
                    "source": person,
                    "target": org.title(),
                    "type": relationship_type,
                    "properties": {}
                })

        # Extract "I love/like X" pattern
        likes_pattern = re.findall(r'i (?:love|like) (\w+)', text_lower)
        for concept in likes_pattern:
            entities.append({"type": "CONCEPT", "name": concept.title(), "properties": {}})
            relationships.append({
                "source": "User",  # Default to User for first person statements
                "target": concept.title(),
                "type": "LIKES",
                "properties": {}
            })

        # Extract "from X" pattern (alternative location format)
        from_location = re.findall(r'from (\w+)', text_lower)
        for place in from_location:
            entities.append({"type": "PLACE", "name": place.title(), "properties": {}})
            if not any(rel["type"] == "LIVES_IN" for rel in relationships):
                relationships.append({
                    "source": "User",
                    "target": place.title(),
                    "type": "LIVES_IN",
                    "properties": {}
                })

        # Remove duplicate entities
        unique_entities = []
        seen_names = set()
        for entity in entities:
            if entity["name"] not in seen_names:
                unique_entities.append(entity)
                seen_names.add(entity["name"])

        return {"entities": unique_entities, "relationships": relationships}

class ConversationMemory:
    """Stores recent conversation context for better responses"""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def add_interaction(self, question: str, answer: str):
        """Add a Q&A pair to memory"""
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": str(logger.handlers[0].formatter.formatTime() if logger.handlers else "")
        })

        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_summary(self) -> str:
        """Get a summary of recent interactions"""
        if not self.history:
            return "No recent context."

        context_lines = []
        for interaction in self.history[-3:]:  # Last 3 interactions
            context_lines.append(f"Q: {interaction['question']}")
            context_lines.append(f"A: {interaction['answer'][:100]}...")

        return "\n".join(context_lines)

class EnhancedKnowledgeGraphService:
    """Enhanced KG service with real-time entity extraction and updates"""

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

        self.entity_extractor = EntityExtractor(self.ollama_base_url)
        logger.info("Enhanced Knowledge Graph Service initialized")

    def extract_and_store_entities(self, text: str, source: str = "conversation") -> Dict:
        """Extract entities from text and store them in the graph"""
        logger.info(f"Extracting entities from: {text[:100]}...")

        extraction_result = self.entity_extractor.extract_entities(text)

        entities_added = 0
        relationships_added = 0

        try:
            # Store entities first - with fuzzy matching
            entity_name_mapping = {}  # Track original name -> standardized name

            for entity in extraction_result.get("entities", []):
                entity_type = entity.get("type", "ENTITY")
                entity_name = entity.get("name", "").strip()
                properties = entity.get("properties", {})
                properties["source"] = source

                if entity_name:
                    # Check for existing similar entities (fuzzy matching)
                    standardized_name = self._find_or_create_entity(entity_name, entity_type, properties)
                    entity_name_mapping[entity_name] = standardized_name
                    entities_added += 1

            # Store relationships with two-step creation
            for rel in extraction_result.get("relationships", []):
                source_name = rel.get("source", "").strip()
                target_name = rel.get("target", "").strip()
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {})
                properties["extracted_from"] = source

                if source_name and target_name:
                    # Use standardized names from mapping, or create entities if missing
                    final_source = entity_name_mapping.get(source_name, source_name)
                    final_target = entity_name_mapping.get(target_name, target_name)

                    # Ensure both entities exist before creating relationship
                    if not final_source in entity_name_mapping.values():
                        final_source = self._find_or_create_entity(source_name, "ENTITY", {"source": source})
                        entity_name_mapping[source_name] = final_source

                    if not final_target in entity_name_mapping.values():
                        final_target = self._find_or_create_entity(target_name, "ENTITY", {"source": source})
                        entity_name_mapping[target_name] = final_target

                    # Create relationship with verified entity names
                    success = self._create_relationship(final_source, final_target, rel_type, properties)
                    if success:
                        relationships_added += 1
                        logger.info(f"Created relationship: {final_source} -{rel_type}-> {final_target}")

        except Exception as e:
            logger.error(f"Error storing entities: {e}", exc_info=True)

        logger.info(f"Added {entities_added} entities and {relationships_added} relationships")
        return {
            "entities_added": entities_added,
            "relationships_added": relationships_added,
            "extraction_result": extraction_result,
            "entity_mapping": entity_name_mapping
        }

    def _find_or_create_entity(self, entity_name: str, entity_type: str, properties: Dict) -> str:
        """Find existing similar entity or create new one, return standardized name"""
        try:
            # First, try exact match
            exact_match_query = """
            MATCH (e)
            WHERE e.name = $name
            RETURN e.name as name, labels(e) as types
            LIMIT 1
            """
            result = self.graph.query(exact_match_query, {"name": entity_name})

            if result:
                return result[0]["name"]

            # Try fuzzy matching (case-insensitive, partial match)
            fuzzy_query = """
            MATCH (e)
            WHERE toLower(e.name) CONTAINS toLower($name)
               OR toLower($name) CONTAINS toLower(e.name)
               OR e.name =~ ('(?i).*' + $name + '.*')
            RETURN e.name as name, labels(e) as types
            LIMIT 1
            """
            result = self.graph.query(fuzzy_query, {"name": entity_name})

            if result:
                logger.info(f"Found similar entity: {entity_name} -> {result[0]['name']}")
                return result[0]["name"]

            # No match found, create new entity
            create_query = f"""
            CREATE (e:{entity_type} {{name: $name}})
            SET e += $properties
            RETURN e.name as name
            """
            result = self.graph.query(create_query, {"name": entity_name, "properties": properties})
            logger.info(f"Created new entity: {entity_name} ({entity_type})")
            return entity_name

        except Exception as e:
            logger.error(f"Error in _find_or_create_entity: {e}")
            return entity_name

    def _create_relationship(self, source_name: str, target_name: str, rel_type: str, properties: Dict) -> bool:
        """Create relationship between two entities, ensuring both exist"""
        try:
            # Verify both entities exist and create relationship
            query = f"""
            MATCH (s), (t)
            WHERE s.name = $source_name AND t.name = $target_name
            MERGE (s)-[r:{rel_type}]->(t)
            SET r += $properties
            RETURN r, s.name as source, t.name as target
            """

            result = self.graph.query(query, {
                "source_name": source_name,
                "target_name": target_name,
                "properties": properties
            })

            if result:
                logger.debug(f"Relationship created: {source_name} -{rel_type}-> {target_name}")
                return True
            else:
                logger.warning(f"Failed to create relationship: {source_name} -{rel_type}-> {target_name} (entities not found)")
                return False

        except Exception as e:
            logger.error(f"Error creating relationship {source_name} -{rel_type}-> {target_name}: {e}")
            return False

    def find_similar_entities(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Find entities similar to the query using text matching"""
        try:
            # Simple text-based similarity search in Neo4j
            cypher_query = """
            MATCH (n)
            WHERE n.name CONTAINS $query_text
               OR $query_text CONTAINS n.name
               OR n.name =~ ('(?i).*' + $query_text + '.*')
            RETURN DISTINCT n.name as name, labels(n) as types, n as entity
            LIMIT $limit
            """

            results = self.graph.query(cypher_query, {
                "query_text": query_text,
                "limit": limit
            })

            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

class EnhancedReasoningService:
    """Enhanced reasoning service with real-time learning and context"""

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

        self.kg_service = EnhancedKnowledgeGraphService()
        self.conversation_memory = ConversationMemory()

        # Refresh schema and initialize chain
        self.graph.refresh_schema()

        # Enhanced QA prompt
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful neuro-symbolic AI assistant that learns from conversations.

Context from knowledge graph:
{context}

Question: {question}

Instructions:
- If context contains "CONVERSATIONAL_QUERY": Respond warmly and conversationally
- If context contains "NO_GRAPH_DATA": Use general knowledge but mention you're learning
- If context contains real data: Use it to provide accurate, helpful answers
- Always be helpful and engaging, and mention that you're learning from our conversation

Helpful Answer:
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

        logger.info("Enhanced Reasoning Service initialized")

    def query(self, question: str) -> str:
        """Enhanced query method with real-time learning"""
        logger.info(f"Processing enhanced query: {question}")

        try:
            # Extract and store entities from the question
            self.kg_service.extract_and_store_entities(question, "user_question")

            # Find similar entities for context
            similar_entities = self.kg_service.find_similar_entities(question)
            similar_context = ""
            if similar_entities:
                similar_names = [entity["name"] for entity in similar_entities[:3]]
                similar_context = f"Related entities in knowledge graph: {', '.join(similar_names)}"

            # Get conversation context
            conversation_context = self.conversation_memory.get_context_summary()

            # Prepare enhanced context for Cypher generation
            enhanced_context = f"{conversation_context}\n{similar_context}"

            # Update the cypher chain with context (only query for cypher generation)
            result = self.cypher_chain.invoke({
                "query": question,
                "context": enhanced_context
            })

            answer = result.get('result', 'I apologize, but I encountered an issue processing your request.')

            # Store the interaction in memory
            self.conversation_memory.add_interaction(question, answer)

            # Extract entities from the answer and store them
            self.kg_service.extract_and_store_entities(answer, "assistant_response")

            logger.info(f"Enhanced query completed successfully")
            return answer

        except Exception as e:
            logger.error(f"Enhanced query failed: {e}")
            return "I apologize, but I encountered an issue processing your request. Please try again."