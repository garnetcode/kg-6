#!/usr/bin/env python
"""
Quick test to check relationships in knowledge graph
"""
import os
import sys
import django

# Add project root to path
sys.path.insert(0, '/Users/shadow/PycharmProjects/kg-6')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'neuro_symbolic_agent.settings')
django.setup()

from rag_agent.enhanced_services import EnhancedKnowledgeGraphService

kg = EnhancedKnowledgeGraphService()

# Query all relationships
query = """
MATCH (a)-[r]->(b)
RETURN a.name as source, type(r) as relationship, b.name as target
ORDER BY a.name
LIMIT 20
"""

results = kg.graph.query(query)
print(f"Relationships in knowledge graph: {len(results)}")
for rel in results:
    print(f"  {rel['source']} --{rel['relationship']}--> {rel['target']}")

# Check specifically for Garnet/Bowspace
garnet_query = """
MATCH (a)-[r]->(b)
WHERE toLower(a.name) CONTAINS 'garnet'
   OR toLower(b.name) CONTAINS 'bowspace'
   OR toLower(a.name) CONTAINS 'user'
RETURN a.name as source, type(r) as relationship, b.name as target
"""

garnet_results = kg.graph.query(garnet_query)
print(f"\nGarnet/Bowspace relationships: {len(garnet_results)}")
for rel in garnet_results:
    print(f"  {rel['source']} --{rel['relationship']}--> {rel['target']}")