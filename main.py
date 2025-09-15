import os
import sqlalchemy
from sqlalchemy import create_engine, text
import boto3
import json
from langchain_aws import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_pinecone import Pinecone

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
FEEDBACK_THRESHOLD = int(os.getenv("FEEDBACK_THRESHOLD", 5))
TEACHER_LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0" # Using Sonnet as the 'teacher'
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-index"

def get_negative_feedback_interactions(engine):
    """Fetches interactions with negative feedback that haven't been processed."""
    with engine.connect() as connection:
        query = text("""
            SELECT interaction_id, user_query 
            FROM interactions 
            WHERE feedback = -1 AND processed_for_training = FALSE
        """)
        result = connection.execute(query)
        return result.fetchall()

def generate_improved_answer(query, llm):
    """Generates a high-quality answer for a given query using the teacher LLM."""
    # Add context to guide the teacher LLM to act as a Flipkart agent
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

def mark_interaction_as_processed(engine, interaction_id):
    """Marks an interaction as processed to avoid reprocessing."""
    with engine.connect() as connection:
        stmt = text("UPDATE interactions SET processed_for_training = TRUE WHERE interaction_id = :id")
        connection.execute(stmt, {"id": interaction_id})
        connection.commit()

def main():
    print("Starting self-healing training job...")
    
    # --- Initialize Clients ---
    engine = create_engine(DATABASE_URL)
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    teacher_llm = BedrockChat(client=bedrock_client, model_id=TEACHER_LLM_MODEL_ID)
    embeddings = BedrockEmbeddings(client=bedrock_client)
    vector_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    # 1. Fetch negative feedback
    interactions = get_negative_feedback_interactions(engine)
    
    print(f"Found {len(interactions)} interactions with negative feedback to process.")

    # 2. Check against threshold
    if len(interactions) < FEEDBACK_THRESHOLD:
        print(f"Number of interactions ({len(interactions)}) is below the threshold ({FEEDBACK_THRESHOLD}). Exiting job.")
        return

    # 3. Process interactions
    for interaction in interactions:
        interaction_id, user_query = interaction
        print(f"Processing interaction {interaction_id} for query: '{user_query}'")

        # 4. Generate improved answer
        improved_answer = generate_improved_answer(user_query, teacher_llm)
        
        if improved_answer:
            # 5. Update knowledge base
            update_knowledge_base(vector_store, user_query, improved_answer)
            
            # 6. Mark as processed
            mark_interaction_as_processed(engine, interaction_id)
            print(f"Successfully processed interaction {interaction_id}.")

    print("Self-healing training job finished.")

if __name__ == "__main__":
    main()
