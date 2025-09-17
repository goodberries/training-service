import os
import boto3
import json
from langchain_aws import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_pinecone import Pinecone
import httpx

# --- Configuration --- //test
INTERACTIONS_SERVICE_URL = os.getenv("INTERACTIONS_SERVICE_URL", "http://interactions-service:8003")
FEEDBACK_THRESHOLD = int(os.getenv("FEEDBACK_THRESHOLD", 5))
TEACHER_LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
JUDGE_LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-index"

def get_negative_feedback_interactions():
    """Fetches interactions with negative feedback via the interactions-service."""
    params = {"feedback": -1, "processed_for_training": "false"}
    try:
        response = httpx.get(f"{INTERACTIONS_SERVICE_URL}/interactions", params=params)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"Error fetching negative feedback from interactions-service: {e.response.text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while fetching interactions: {e}")
        return []

def has_sufficient_existing_answer(vector_store, judge_llm, query, k=3):
    """Checks if existing KB sufficiently answers the query using similarity + LLM judge."""
    try:
        docs = vector_store.similarity_search(query, k=k)
    except Exception as e:
        print(f"Similarity search failed: {e}")
        return False
    if not docs:
        return False
    context = "\n\n".join([f"Doc {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
    judge_prompt = (
        "You are validating if the provided context already contains an adequate answer for the user's question in the Flipkart support domain. "
        "Answer strictly with 'yes' or 'no'.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nDoes the context adequately answer the question?"
    )
    try:
        verdict = judge_llm.invoke(judge_prompt).content.strip().lower()
        return verdict.startswith('y')
    except Exception as e:
        print(f"LLM judge failed: {e}")
        return False

def generate_improved_answer(query, llm):
    """Generates a high-quality answer for a given query using the teacher LLM."""
    context_prompt = (
        "You are an expert customer support agent for Flipkart, an e-commerce company. "
        "Your goal is to provide a clear, helpful, and factually correct answer based on company policy. "
        "A user was unsatisfied with a previous answer to the following question. "
        "Please provide the ideal, comprehensive, and correct answer that a Flipkart agent should give.\n\n"
        f"User Question: {query}"
    )
    try:
        response = llm.invoke(context_prompt)
        return response.content
    except Exception as e:
        print(f"Error generating improved answer for query '{query}': {e}")
        return None

def update_knowledge_base(vector_store, query, improved_answer):
    """Embeds and upserts the new Q&A pair into Pinecone."""
    try:
        vector_store.add_texts(
            [f"Question: {query}\nAnswer: {improved_answer}"],
            metadatas=[{"source": "self-healing-feedback"}]
        )
        print(f"Successfully updated knowledge base for query: {query}")
    except Exception as e:
        print(f"Error updating Pinecone for query '{query}': {e}")

def mark_interaction_as_processed(interaction_id):
    """Marks an interaction as processed via the interactions-service."""
    try:
        url = f"{INTERACTIONS_SERVICE_URL}/interactions/{interaction_id}/processed"
        response = httpx.patch(url, json={"processed_for_training": True})
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"Error marking interaction {interaction_id} as processed: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred while marking interaction {interaction_id} as processed: {e}")

def main():
    print("Starting self-healing training job...")
    
    # --- Initialize Clients ---
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    teacher_llm = BedrockChat(client=bedrock_client, model_id=TEACHER_LLM_MODEL_ID)
    judge_llm = BedrockChat(client=bedrock_client, model_id=JUDGE_LLM_MODEL_ID)
    embeddings = BedrockEmbeddings(client=bedrock_client)
    vector_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    # 1. Fetch negative feedback
    interactions = get_negative_feedback_interactions()
    
    print(f"Found {len(interactions)} interactions with negative feedback to process.")

    # 2. Check against threshold
    if len(interactions) < FEEDBACK_THRESHOLD:
        print(f"Number of interactions ({len(interactions)}) is below the threshold ({FEEDBACK_THRESHOLD}). Exiting job.")
        return

    # 3. Process interactions
    for interaction in interactions:
        interaction_id = interaction['interaction_id']
        user_query = interaction['user_query']
        print(f"Processing interaction {interaction_id} for query: '{user_query}'")

        # 3a. Check if existing KB already has a sufficient answer
        if has_sufficient_existing_answer(vector_store, judge_llm, user_query, k=3):
            print("Existing knowledge is sufficient. Skipping generation and marking as processed.")
            mark_interaction_as_processed(interaction_id)
            continue

        # 4. Generate improved answer
        improved_answer = generate_improved_answer(user_query, teacher_llm)
        
        if improved_answer:
            # 5. Update knowledge base
            update_knowledge_base(vector_store, user_query, improved_answer)
            
            # 6. Mark as processed
            mark_interaction_as_processed(interaction_id)
            print(f"Successfully processed interaction {interaction_id}.")

    print("Self-healing training job finished.")

if __name__ == "__main__":
    main()
