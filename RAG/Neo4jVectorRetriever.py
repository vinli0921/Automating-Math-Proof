import openai
import numpy as np
import json
from typing import Any
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

class Neo4jVectorRetriever(BaseRetriever):
    driver: Any
    top_k: int = 5

    def get_query_embedding(self, query):
        response = openai.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
        return query_embedding
    
    def structure_node_content(self, node):
        """Structure mathematical content from node."""
        content_parts = []
        
        if node.get('title'):
            content_parts.append(f"Title: {node['title']}")
        
        if node.get('theorem'):
            content_parts.append(f"Theorem: {node['theorem']}")
        
        if node.get('proof'):
            content_parts.append(f"Proof: {node['proof']}")
            
        try:
            if node.get('math_expressions'):
                expressions = json.loads(node['math_expressions'])
                if expressions:
                    content_parts.append(f"Mathematical Expressions: {' '.join(expressions)}")
        except json.JSONDecodeError:
            pass
            
        return "\n\n".join(content_parts)

    def get_top_node(self, query_embedding):
        with self.driver.session() as session:
            results = session.read_transaction(
                self._similarity_search, query_embedding, top_k=1)
        if results:
            record = results[0]
            node = record['node']
            content = self.structure_node_content(node)
            node_id = node.get('id')
            return node_id, content
        else:
            return None, None

    def get_neighbors(self, node_id, top_k):
        with self.driver.session() as session:
            results = session.read_transaction(
                self._get_neighbors_tx, node_id, top_k
            )
        # Structure the content for each neighbor
        structured_results = []
        for result in results:
            result['content'] = self.structure_node_content(result)
            structured_results.append(result)
        return structured_results

    def _get_relevant_documents(self, query):
        query_embedding = self.get_query_embedding(query)
        with self.driver.session() as session:
            results = session.read_transaction(
                self._similarity_search, query_embedding, self.top_k)
        documents = []
        for record in results:
            node = record['node']
            content = self.structure_node_content(node)
            metadata = {
                'id': node.get('id'),
                'similarity': record.get('score'),
                'type': node.get('type')
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        return documents

    @staticmethod
    def _similarity_search(tx, query_embedding, top_k=10):
        result = tx.run("""
        WITH $embedding AS query_embedding
        CALL db.index.vector.queryNodes(
            'nodeContent',
            $top_k,
            query_embedding
        ) YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """, embedding=query_embedding, top_k=top_k)
        
        nodes = [record.data() for record in result]
        return nodes

    @staticmethod
    def _get_neighbors_tx(tx, node_id, top_k):
        # Updated to use relationship types and mathematical structure
        result = tx.run("""
        MATCH (n)-[r:LINK]->(neighbor)
        WHERE n.id = $node_id
        WITH neighbor, r.type AS rel_type, r.context AS context,
             CASE r.type 
                WHEN 'USES_DEFINITION' THEN 5
                WHEN 'USES_THEOREM' THEN 4
                WHEN 'PROOF_TECHNIQUE' THEN 3
                WHEN 'SIMILAR_PROOF' THEN 2
                ELSE 1
             END as type_weight
        RETURN 
            neighbor.id AS id,
            neighbor.title AS title,
            neighbor.theorem AS theorem,
            neighbor.proof AS proof,
            neighbor.math_expressions AS math_expressions,
            neighbor.plotEmbedding AS embedding,
            rel_type,
            context,
            type_weight
        ORDER BY type_weight DESC
        LIMIT $top_k
        """, node_id=node_id, top_k=top_k)
        
        neighbors = [record.data() for record in result]
        return neighbors
