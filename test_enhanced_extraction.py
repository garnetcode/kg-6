#!/usr/bin/env python
"""
Direct test of enhanced relationship extraction functionality
"""
import os
import sys
import django

# Add project root to path
sys.path.insert(0, '/Users/shadow/PycharmProjects/kg-6')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'neuro_symbolic_agent.settings')

# Configure Django
django.setup()

from rag_agent.enhanced_services import EnhancedReasoningService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_relationship_extraction():
    """Test the enhanced relationship extraction with complex scenarios"""
    print("🧪 Testing Enhanced Relationship Extraction")
    print("=" * 50)

    try:
        # Initialize the enhanced reasoning service
        print("Initializing Enhanced Reasoning Service...")
        service = EnhancedReasoningService()

        # Test scenarios
        test_cases = [
            "Hi, I am Sarah from New York and I work at Google. I love pizza and programming.",
            "I know John who lives in London and works at Microsoft. He likes AI.",
            "Alice from Paris works at Apple and loves music. She knows Bob from Tokyo."
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}: {test_case}")
            print("-" * 40)

            # Process the query (this will extract entities and create relationships)
            response = service.query(test_case)
            print(f"Response: {response}")

        # Query the knowledge graph to see what was stored
        print("\n📊 Knowledge Graph Summary:")
        print("-" * 30)

        # Get all entities
        entities_query = """
        MATCH (n)
        RETURN labels(n) as types, n.name as name, n.source as source
        ORDER BY n.name
        """
        entities = service.graph.query(entities_query)
        print(f"Entities found: {len(entities)}")
        for entity in entities:
            print(f"  - {entity['name']} ({entity['types']}) [source: {entity['source']}]")

        # Get all relationships
        relationships_query = """
        MATCH (a)-[r]->(b)
        RETURN a.name as source, type(r) as relationship, b.name as target, r.extracted_from as source_context
        ORDER BY a.name, type(r), b.name
        """
        relationships = service.graph.query(relationships_query)
        print(f"\nRelationships found: {len(relationships)}")
        for rel in relationships:
            print(f"  - {rel['source']} --{rel['relationship']}--> {rel['target']} [from: {rel['source_context']}]")

        if relationships:
            print("\n✅ SUCCESS: Relationships are being extracted and stored!")
        else:
            print("\n❌ ISSUE: No relationships found in the knowledge graph")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_relationship_extraction()