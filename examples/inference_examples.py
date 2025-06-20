from reranker import QueryReranker

model_path = "/path/to/model"
reranker = QueryReranker(model_path=model_path)

# Test data
query = "What is the financial benefit of the partnership?"
image_list = [
    "/path/to/images/image1.png", 
    "/path/to/images/image2.png", 
    "/path/to/images/image3.png", 
    "/path/to/images/image4.png", 
    "/path/to/images/image5.png"
]

# Execute reranking
predicted_order = reranker.rerank(query, image_list)

print(f"Query: {query}")
print(f"Reranked order: {predicted_order}")
