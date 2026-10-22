# 3) Conditional workflow
# So let's say there are 3 branches in our college: BCA, BBA and B.COM.
# A student will choose one of these 3 options, and then a chatbot will be activated. 
# You can ask any question to that chatbot, but the LLM you are using in that chatbot does not have knowledge of the college programme.
# For that, you will have 2 PDFs: the first one will be for academics, and the second one will be for fee-related things.
# So technically, we will have 3 conditional paths:

#1 Answering the question using the academic PDF (using RAG)
# 2Answering the question using the fee PDF (using RAG)
# 3Answering general questions based on the LLM's own knowledge

# And every conditional path's response converges to a single node.



import os
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm=ChatGroq(model="openai/gpt-oss-120b",temperature=1.5)

def build_retriver(pdf_path:str):
    loader=PyPDFLoader(pdf_path)
    document=loader.load()
    
    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    chunks=splitter.split_documents(document)

    vectorestore=FAISS.from_documents(chunks,embedding)

    return vectorestore.as_retriever(search_kwargs={"k":4})

academic_retriever=build_retriver("academics_handbook.pdf")
fee_retriver=build_retriver("fee_structure.pdf")

#state
class State(TypedDict):
    programme:str
    message:Annotated[list,add_messages] #history maintain karega add_messages
    query_type:str
    retrieved_context:str

#step -Nodes generation

def classifier_node(state:State)->dict:
    """ Look at the latest user message and decide which path to take"""
    last_message=state['message'][-1].content
    prompt=(
        "classify the followig student query into exactly one category"
        "'acedmic','fee', or 'general'.\n\n"
        "promotion course structure, summer training,or degree requirements.\n"
        "Use 'fee' for question about tution,payment,refund,late charges,'schlarship,or any money-related topic.\n"
        "Use 'general' for greetings ,casual talk or anything not realted to the college rules or fees"
        f"Query :{last_message}\n\n"
        "Return only one word :academic ,fee,or general"
    )
    response=llm.invoke(prompt)
    category=response.content.strip().lower()

    if "academic" in category:
        category="academic"
    elif "fee" in category:
        category="fee"
    else:
        category="general"

    return {"query_type":category}

def academic_rag_node(state:State)->dict:
    """Rtrive relevant chunks from the acedmics handbook"""
    query=state['message'][-1].content
    docs=academic_retriever.invoke(query)
    context="\n\n".join(doc.page_content for doc in docs)

    return {"retrieved_context":context}

def fee_rag_node(state:State)->dict:
    """Retrive relevant chunks from the fee structure PDF"""
    query=state['message'][-1].content
    docs=fee_retriver.invoke(query)
    context="\n\n".join(doc.page_content for doc in docs)

    return {"retrieved_context":context}

def general_node(state:State)->dict:
    """Answer directly using LLM's own knowledge ,no retrieval needed ."""
    
    return {"retrieved_context":"NO_RETRIEVAL_NEEDED"}


def response_node(state:State)->dict:
    """Generates the final; answer ,pretrained using the students's programme."""
    query=state['message'][-1].content
    programme=state.get("programme","UNKNOWN")
    context=state["retrieved_context"]

    if context=="NO_RETRIEVAL_NEEDED":
        prompt=(
            f"You are a friendly college assistant talking to a {programme} student"
            f"Answer this question using your own genearal knowledge:{query}"
        )

    else:
        prompt=(
            f"You are college assistant helping a {programme} student"
            f"Use the following context from the official college document"
            f"The question accurately .If the context mentions specific "
            f"Different programme,highlight the one relevant to{programme} if possible."
            f"Context:\n{context}\n\n"
            f"Question:{query}\n\n"
            f"Give a clear,friendly and precise answer"
        )
    response=llm.invoke(prompt)

    return {
        "message": [AIMessage(content=response.content.strip())]
    }

#step 4 -route function

def route_query(state:State):
    if state['query_type']=='academic':
        return "academic_rag"
    elif state['query_type']=='fee':
        return "fee_rag"
    else:
        return "general"

#step 5: Graph
graph=StateGraph(State)
graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)

#Step 6:Edges
graph.add_edge(START,"classifier")
graph.add_conditional_edges("classifier",route_query)
graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")
graph.add_edge("response",END)

app=graph.compile()

print("Welcome ! to the college assistant \n\n")
print("1.BCA")
print("2.BBA")
print("3.B.com(H)")

choice=input("Enter 1 or 2 or 3")

programme_map={
    "1":"BCA",
    "2":"BBA",
    "3":"B.com(H)"
}

student_programme=programme_map.get(choice,"BCA")

print(f"\n Greet! You are set as a {student_programme} student")


while True:
    user_query=input("----YOU-----")

    if user_query in ["exit","quit"]:
        break
    result=app.invoke({
        "programme":student_programme,
        "message":[("human",user_query)]
    })

    print(f"Assistant :{result['message'][-1].content}")



