from neo4j import GraphDatabase
import numpy as np
from sklearn.cluster import KMeans

# Neo4j connection details
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"

# Function to run queries on Neo4j
def run_query(driver, query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return result.data()

# Initialize Neo4j driver
driver = GraphDatabase.driver(uri, auth=(username, password))

# Step 1: Generate Node Embeddings with Node2Vec
node2vec_parameters = {
    'embeddingDimension': 128,  # Adjusted back to 128 to capture more features
    'walkLength': 80,
    'iterations': 10,
    'writeProperty': 'embedding'
}

node2vec_query = """
CALL gds.node2vec.stream('myGraph', {embeddingDimension: 128})
YIELD nodeId, embedding
RETURN nodeId, embedding
"""

# Run Node2Vec query and get embeddings
node2vec_result = run_query(driver, node2vec_query, node2vec_parameters)

# Extract node IDs and embeddings
node_ids = [record['nodeId'] for record in node2vec_result]
embeddings = np.array([record['embedding'] for record in node2vec_result])

# Print Node2Vec Result
print("Node2Vec Result:", node2vec_result)

# Step 2: Apply K-Means Clustering
# Define the number of communities (clusters)
num_clusters = 5

# Apply K-Means clustering to the embeddings
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)

# Get the cluster labels (community assignments)
communities = kmeans.labels_

# Print K-Means Clustering Result
print("K-Means Clustering Result:", communities)

# Step 3: Assign Communities Back to Nodes in Neo4j
# Prepare community assignments for Neo4j
community_assignments = [{"nodeId": node_id, "communityId": int(community_id)} for node_id, community_id in zip(node_ids, communities)]

# Cypher query to assign community labels to nodes
assign_communities_query = """
UNWIND $communityAssignments AS assignment
MATCH (n)
WHERE ID(n) = assignment.nodeId
SET n.community = assignment.communityId
"""

# Run the query to assign communities
run_query(driver, assign_communities_query, {"communityAssignments": community_assignments})

# Print message indicating completion
print("Communities have been assigned to the nodes in Neo4j.")

# Close the Neo4j driver connection
driver.close()
