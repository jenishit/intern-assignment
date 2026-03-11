This is a FastAPI backend for a conversational AI agent that dynamically uses tools and RAG to answer user queries
It has only one end point POST /chat in which the user enter and queries and the Agent asnwers accordingly.

Also this is only a backend service. So you have to run the backend and go to 
http://127.0.0.1:8000/docs to play with the API and QUestion it.
or You can do
http POST http://localhost:8000/chat query="Your query" #For this you have to install httpie


Agent's Decision Tree:
First it analyses the query. If it contains 'order' and 'digits' with or without '#' then it extracts the order id. The order details is fetched from the tools. The query is again analyzed for any words that would trigger the use of Policy information. 
If there is a policy related string in the query then the RAG is checked else it returns the order information only.
If there is no order information or policy related question then the model converses as a normal AI Agent. The decision parameter is set in "agent.py"
`
def run_agent(
        query: str,
) -> dict:
    order_id_match = re.search(r"#\s*(\d+)", query) or re.search(r"order\s+.*?(\d{4,})", query, re.IGNORECASE)

    if order_id_match:
        order_id = order_id_match.group(1)

        order_data = get_order_details(order_id)

        if "error" not in order_data and _needs_policy(query):
            customer_type = order_data.get("customer_type", "Standard")
            policy_data = retrieve_policy(f"{customer_type} return window")

        elif "error" not in order_data:
            messages.append({
                "role": "user",
                "content": (
                    f"{query}\n\n"
                    f"Tool Result:\n"
                    f"Order Data: {json.dumps(order_data, indent=2)}\n\n"
                    f"Answer concisely using ONLY this tool result."
                ),
            })
        else:
            # Order not found
            messages.append({
                "role": "user",
                "content": (
                    f"{query}\n\n"
                    f"Tool Result:\n"
                    f"{json.dumps(order_data)}\n\n"
                    f"Report that the order was not found."
                ),
            })
    elif _needs_policy(query):
        policy_data = retrieve_policy(query)

    else:
        # No order ID and not a policy question — allow general conversation
        messages.append({
            "role": "user",
            "content": query,
        })
`


If the policy document was 10,000 pages, how would you change your RAG approach?
-> Currently I have used Term Frequency and Inverse Document Frequency for the RAG search. If the policy document was 10,000 pages, there are different ways that we can deal with.
a. Semantic search
    We can use embeddings for better relevance understanding of the document. The langchain_community package provides the methods as well as Vector store FAISS
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = OllamaEmbeddings(model="gemma3:4b")
        vector_store = FAISS.from_documents(documents, embeddings)

b. Vector Database
    We can use vector database liek Pinecone for this. Using vector database it is faster and easier to locate the relevant informations

c. RAPTOR
    I heard about this method recently. It can be used for a RAG system with large chunks of data. What this does is, it makes a hierarchical tree of the document bottom-up instead of one flat list of chunks that other methods do.

If this API served hundreds of users at once, what bottlenecks exist in your code and how would you fix them?
-> Currently, my code can serve one user at a time. To to fix this bottleneck, I would use:
a. Asynchronous LLM inference like

`import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=5)
loop = asyncio.get_event_loop()

async def run_agent_async(query: str):
    result = await loop.run_in_executor(executor, run_agent, query)
    return result

@app.post("/chat")
async def chat(request: ChatRequest):
    result = await run_agent_async(request.query)
    return ChatResponse(...)`

b. Instead of using dictionary i.e. ORDERS_DB = {...} I would use a real database that too a vector database.

c. Currently even if I ask same question repeatedly, the agent infers it to the agent and fetches the result from base everytime. I should incorporate caching, so that if multiple user asks the same question, the agent no longer has to infer it to the LLM and answer from the cache. Not only it increases scalability, but also decreases the laods on the LLM and decreases the billing amount.

d. Rightnow I am using uvicorn and it has only 1 worker. I can increase the worker number to make it a multiworker. like:
`python -m uvicorn main:app --workers 4 `
This handles 4x concurrency