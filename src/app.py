import logging
from enum import Enum

import boto3
import pymongo
from langchain.agents import AgentType, initialize_agent
from langchain.embeddings import BedrockEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools import StructuredTool
from langchain_community.chat_models import BedrockChat
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pydantic import BaseModel, Field

import utils


# Define Your Models and Parameters
class ModelId(str, Enum):
    CLAUDE_3_H = "anthropic.claude-3-haiku-20240307-v1:0"
    AWS_Titan_Embed_Text = "amazon.titan-embed-text-v1"


class ModelKwargs(BaseModel):
    temperature: float = Field(default=0.5, ge=0, le=1)
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    top_p: float = Field(default=0.5, ge=0, le=1)
    top_k: int = Field(default=0, ge=0, le=500)
    stop_sequences: list = Field(["\n\nHuman"])


llm_model = ModelId.CLAUDE_3_H.value
embedding_model_id = ModelId.AWS_Titan_Embed_Text.value

field_name_to_be_vectorized = "About Place"
vector_field_name = "details_embedding"
index_name = "awsanthropic_vector_index"


# Setup bedrock
def setup_bedrock():
    """Initialize the Bedrock runtime."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )


def initialize_llm(client):
    """Initialize the language model."""
    llm = BedrockChat(client=client, model_id=llm_model)
    llm.model_kwargs = ModelKwargs().__dict__
    return llm


bedrock_runtime = setup_bedrock()

# Connect to the MongoDB database
mongoDBClient = pymongo.MongoClient(utils.get_MongoDB_Uri())
logging.info("Connected to MongoDB...")

database = mongoDBClient["anthropic-travel-agency"]
collection = database["trip_recommendations"]


def mongodb_place_lookup_by_country(query_str: str) -> str:
    """
    Retrieve places by country name from the MongoDB database.

    This function connects to the MongoDB Atlas database and searches for place names
    that match the provided country name query. The search is case-insensitive and
    utilizes a regex match. The function returns a list of matching place names.

    Args:
        query_str (str): The country name to search for. It may include partial matches.

    Returns:
        str: A string representation of the list of place names that match the country query.
    """
    res = ""
    res = collection.aggregate(
        [
            {"$match": {"Country": {"$regex": query_str, "$options": "i"}}},
            {"$project": {"Place Name": 1}},
        ]
    )
    places = []
    for place in res:
        places.append(place["Place Name"])
    return str(places)


def mongodb_place_lookup_by_name(query_str: str) -> str:
    """
    Retrieve place information by place name from the MongoDB database.

    This function connects to the MongoDB Atlas database and searches for a place
    using the provided name query. The search checks both the place name and country,
    using case-insensitive regex matching. The function returns the first matching
    document without the ID field.

    Args:
        query_str (str): The place name to search for. It may include partial matches.

    Returns:
        str: A string representation of the first matching document's details without the ID field.
        """
    res = ""
    filter = {
        "$or": [
            {"Place Name": {"$regex": query_str, "$options": "i"}},
            {"Country": {"$regex": query_str, "$options": "i"}},
        ]
    }
    project = {"_id": 0,"details_embedding":0}

    res = collection.find_one(filter=filter, projection=project)
    return str(res)


def mongodb_place_lookup_by_best_time_to_visit(query_str: str) -> str:
    """
    Retrieve place information by the best time to visit from the MongoDB database.

    This function connects to the MongoDB Atlas database and searches for the best time
    to visit places using the provided query string. The search is performed on both
    place names and countries using case-insensitive regex matching. The function returns
    the best time to visit for the first matching document.

    Args:
        query_str (str): The place name or country to search for. It may include partial matches.

    Returns:
        str: A string representation of the best time to visit for the first matching document.
    """
    res = ""
    filter = {
        "$or": [
            {"Place Name": {"$regex": query_str, "$options": "i"}},
            {"Country": {"$regex": query_str, "$options": "i"}},
        ]
    }
    project = {"Best Time To Visit": 1, "_id": 0,"details_embedding":0}

    res = collection.find_one(filter=filter, projection=project)
    return str(res)


# filter the data using the criteria and do a Schematic search
def mongodb_search(query: str) -> str:
    """
    Performs a vector search using MongoDB Atlas, retrieving documents based on the semantic similarity of
    their embeddings to the given query.

    This function leverages MongoDB Atlas' vector search capabilities to find documents whose embeddings
    are similar to the query. It supports searching and retrieves documents based on the cosine similarity of their vector representations.

    Args:
        query (str): The search query string to perform the vector search.

    Returns:
        str: A list of document contents retrieved by the vector search, with each document's content
            represented as a string.

    """

    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id=embedding_model_id,
    )
    text_as_embeddings = embeddings.embed_documents([query])
    embedding_value = text_as_embeddings[0]

    # get the vector search results based on the filter conditions.
    response = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "index": "awsanthropic_vector_index",
                    "path": "details_embedding",
                    "queryVector": embedding_value,
                    "numCandidates": 200,
                    "limit": 10,
                }
            },
            {
                "$project": {
                    "score": {"$meta": "searchScore"},
                    field_name_to_be_vectorized: 1,
                    "_id": 0,
                }
            },
        ]
    )

    # Result is a list of docs with the array fields
    docs = list(response)

    # Extract an array field from the docs
    array_field = [doc[field_name_to_be_vectorized] for doc in docs]

    # Join array elements into a string
    llm_input_text = "\n \n".join(str(elem) for elem in array_field)

    # utility
    newline, bold, unbold = "\n", "\033[1m", "\033[0m"
    logging.info(
        newline
        + bold
        + "Given Input From MongoDB Vector Search: "
        + unbold
        + newline
        + llm_input_text
        + newline
    )

    return llm_input_text


def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        utils.get_MongoDB_Uri(),
        session_id,
        database_name="anthropic-travel-agency",
        collection_name="chat_history",
    )


def interact_with_agent(sessionId, input_query, chat_history):
    """Interact with the agent and store chat history. Return the response."""

    # Initialize bedrock and llm
    bedrock_runtime = setup_bedrock()
    llm = initialize_llm(bedrock_runtime)

    # Initialize tools
    tool_mongodb_search = StructuredTool.from_function(mongodb_search)
    tool_mongodb_place_lookup_by_country = StructuredTool.from_function(
        mongodb_place_lookup_by_country
    )
    tool_mongodb_place_lookup_by_name = StructuredTool.from_function(
        mongodb_place_lookup_by_name
    )
    tool_mongodb_place_lookup_by_best_time_to_visit = StructuredTool.from_function(
        mongodb_place_lookup_by_best_time_to_visit
    )

    tools = [
        tool_mongodb_search,
        tool_mongodb_place_lookup_by_country,
        tool_mongodb_place_lookup_by_name,
        tool_mongodb_place_lookup_by_best_time_to_visit,
    ]

    chat_message_int = MessagesPlaceholder(variable_name="chat_history")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=get_session_history(sessionId),
        return_messages=True,
    )

    PREFIX = """You are a highly capable, friendly, and knowledgeable assistant with access to various tools to extract information from MongoDB databases about any data.
            Engage with the user in a conversational manner and aim to provide accurate and helpful responses to their questions. Respond well to greetings.
            Utilize the all the available tools effectively to gather information, and refer to the chat history as needed. If the chat history shows that you didi not have information earlier that does not mean you do not have it now. Uses the tools again and find the information.
            You have access to the following tools:"""
    FORMAT_INSTRUCTIONS = """Your final response should be in detail and written in clear, human-readable text, free from any extra special characters. Always return only the final answer to the original input question.
            At the end of your response, briefly list the tools you used to arrive at your answer."""
    SUFFIX = """Begin!"""

    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "memory_prompts": [chat_message_int],
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
            "prefix":PREFIX,
            "format_instructions":FORMAT_INSTRUCTIONS,
            "suffix":SUFFIX
        },
        memory=memory,
        verbose=True,
        max_iterations=3
    )
    result = agent_executor.invoke(
        {
            "input": input_query,
            "chat_history": chat_history,
        }
    )
    chat_history.append(HumanMessage(content=input_query))
    chat_history.append(AIMessage(content="Assistant: " + result["output"]))
    return result


def lex_response(res):

    response = {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {"name": "FallbackIntent", "state": "Fulfilled"},
        },
        "messages": [{"contentType": "PlainText", "content": str(res)}],
    }
    return response


def lambda_handler(event, context):

    sessionId = event["sessionId"]
    chat_history = []
    input_text = event["inputTranscript"]
    logging.info(input_text)
    response = interact_with_agent(sessionId, input_text, chat_history)
    logging.info(response["output"])
    response = lex_response(response["output"])
    return response
