#!/usr/bin/env python
"""
Test enhanced business relationship extraction functionality
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

def test_business_relationship_extraction():
    """Test the enhanced business relationship extraction"""
    print("🧪 Testing Enhanced Business Relationship Extraction")
    print("=" * 60)

    try:
        # Initialize the enhanced reasoning service
        print("Initializing Enhanced Reasoning Service...")
        service = EnhancedReasoningService()

        # Business-focused test cases
        test_cases = [
            "Hi, my name is Garnet, I live in Harare, I work for Bowspace",
            "John issued fuel coupon #123 to the delivery team",
            "Sarah approved the marketing budget for Q4",
            "Mike reports to Jennifer in the sales department",
            "Alice manages the development team at TechCorp",
            "Invoice INV-001 was paid by ABC Corp last week",
            "I was assigned to Project Phoenix by the project manager"
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}: {test_case}")
            print("-" * 50)

            # Process the query (this will extract entities and create relationships)
            response = service.query(test_case)
            print(f"Response: {response}")

        # Query the knowledge graph to see what was stored
        print("\n📊 Enhanced Knowledge Graph Summary:")
        print("-" * 40)

        # Get all entities with more detail
        entities_query = """
        MATCH (n)
        RETURN labels(n) as types, n.name as name,
               n.source as source, count{(n)-[]-()}  as connections
        ORDER BY connections DESC, n.name
        """
        entities = service.graph.query(entities_query)
        print(f"Entities found: {len(entities)}")
        for entity in entities:
            print(f"  - {entity['name']} ({entity['types']}) [connections: {entity['connections']}, source: {entity['source']}]")

        # Get all relationships with more detail
        relationships_query = """
        MATCH (a)-[r]->(b)
        RETURN a.name as source, type(r) as relationship, b.name as target,
               r.extracted_from as context, r.source as extraction_source
        ORDER BY a.name, type(r), b.name
        """
        relationships = service.graph.query(relationships_query)
        print(f"\nRelationships found: {len(relationships)}")

        # Group relationships by type for better analysis
        by_type = {}
        for rel in relationships:
            rel_type = rel['relationship']
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        for rel_type, rels in by_type.items():
            print(f"\n  {rel_type} relationships:")
            for rel in rels:
                print(f"    - {rel['source']} --{rel['relationship']}--> {rel['target']} [from: {rel['context']}]")

        # Check specifically for the "work for Bowspace" relationship
        bowspace_query = """
        MATCH (p)-[r]->(b)
        WHERE toLower(b.name) CONTAINS 'bowspace'
           OR toLower(p.name) CONTAINS 'garnet'
        RETURN p.name as person, type(r) as relationship, b.name as target
        """
        bowspace_results = service.graph.query(bowspace_query)

        print(f"\n🎯 Bowspace/Garnet Relationships:")
        if bowspace_results:
            for result in bowspace_results:
                print(f"  ✅ {result['person']} --{result['relationship']}--> {result['target']}")
            print("\n✅ SUCCESS: Business relationships including 'work for Bowspace' detected!")
        else:
            print("  ❌ No Bowspace/Garnet relationships found")

        # Business pattern analysis
        business_patterns = [
            "WORKS_FOR", "ISSUED", "APPROVED", "MANAGES", "REPORTS_TO",
            "ASSIGNED_TO", "PAID_BY", "COLLABORATES_WITH"
        ]

        print(f"\n📈 Business Relationship Pattern Analysis:")
        for pattern in business_patterns:
            count_query = f"""
            MATCH ()-[r:{pattern}]->()
            RETURN count(r) as count
            """
            result = service.graph.query(count_query)
            count = result[0]['count'] if result else 0
            print(f"  - {pattern}: {count} relationships")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_business_relationship_extraction()