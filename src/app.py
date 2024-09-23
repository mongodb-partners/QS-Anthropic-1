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
from pydantic import BaseModel, Field
import utils

# Define Your Models and Parameters
class ModelId(str, Enum):
    CLAUDE_3_5_S = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_3_H = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_S = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_O = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_2_1 = "anthropic.claude-v2:1"
    CLAUDE_2 = "anthropic.claude-v2"
    CLAUDE_1 = "anthropic.claude-instant-v1"
    AWS_Titan_Embed_Text = "amazon.titan-embed-text-v1"


class ModelKwargs(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    top_p: float = Field(default=0.999, ge=0, le=1)
    top_k: int = Field(default=0, ge=0, le=500)
    stop_sequences: list = Field(["\n\nHuman"])


class MongoDBConnection:
    def getMongoDBUri():
        mongo_uri = utils.get_secret("workshop/atlas_secret")
        return mongo_uri


llm_model = ModelId.CLAUDE_2_1.value
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
mongoDBClient = pymongo.MongoClient(MongoDBConnection.getMongoDBUri())
logging.info("Connected to MongoDB...")

database = mongoDBClient["anthropic-travel-agency"]
collection = database["trip_recommendations"]


def mongodb_search(query: str) -> str:
    """Returns results from MongoDB related to the user input. Pass text input only."""
    res = mdb_query(query)
    return str(res)


def mongodb_place_lookup_by_country(query_str: str) -> str:
    """Retrieve place by Country Name"""
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
    """Retrieve place by Place Name"""
    res = ""
    filter = {
        "$or": [
            {"Place Name": {"$regex": query_str, "$options": "i"}},
            {"Country": {"$regex": query_str, "$options": "i"}},
        ]
    }
    project = {"_id": 0}

    res = collection.find_one(filter=filter, projection=project)
    return str(res)


def mongodb_place_lookup_by_best_time_to_visit(query_str: str) -> str:
    """Retrieve place by Best Time to Visit"""
    res = ""
    filter = {
        "$or": [
            {"Place Name": {"$regex": query_str, "$options": "i"}},
            {"Country": {"$regex": query_str, "$options": "i"}},
        ]
    }
    project = {"Best Time To Visit": 1, "_id": 0}

    res = collection.find_one(filter=filter, projection=project)
    return str(res)


# filter the data using the criteria and do a schematic search
def mdb_query(query):
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
                    "title": 1,
                    "claim_id": 1,
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


def setup_full_agent():
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
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "memory_prompts": [chat_message_int],
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
        },
        memory=memory,
        verbose=False,
    )

    return agent_executor


def interact_with_agent(agent_executor, input_query, chat_history):
    """Interact with the agent and store chat history. Return the response."""
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

    # Initialize agent
    agent_executor = setup_full_agent()

    chat_history = []
    input_text = event["inputTranscript"]
    response = interact_with_agent(agent_executor, input_text, chat_history)
    logging.info(response["output"])
    response = lex_response(response["output"])
    return response
