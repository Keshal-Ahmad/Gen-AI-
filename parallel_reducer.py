import os
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0.6)

#REDUCERS ,score overwrite na ho isliye ye banega
def merge_score_dicts(existing:dict,newupdate:dict)->dict:
    if existing is None: #agar koi bhi pichla score nahi hai
        return newupdate
    return {**existing,**newupdate}

#create a state
class AnalyzerState(TypedDict):
    raw_input:str
    safety_score:Annotated[dict[str,int],merge_score_dicts]

#creating nodes

def toxicity_node(state: AnalyzerState) -> dict:

    print("\n[Branch 1] Analyzing Toxicity and Hate Speech")

    prompt = (
        """Analyze the following text for toxicity and hate speech.
        Give a toxicity score from 0 to 10,
        where 0 = completely non-toxic and 10 = extremely toxic.
        Return ONLY the integer score from 0 to 10.

        Text:
        """
        f"{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except:
        score = 0

    return {"safety_score": {"toxicity_level": score}}

def copyright_node(state: AnalyzerState) -> dict:

    print("\n[Branch 2] Analyzing Copyright & Originality Risks")

    prompt = (
        """Analyze the following text for copyright and originality risks.
        Give a risk score from 0 to 10,
        where 0 = completely original and 10 = very high copyright/unoriginality risk.
        Return ONLY the integer score from 0 to 10.

        Text:
        """
        f"{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except:
        score = 0

    return {"safety_score": {"copyright_risk": score}}

def cultural_sensitivity_node(state: AnalyzerState) -> dict:

    print("\n[Branch 3] Analyzing Regional & Cultural Sensitivity")

    prompt = (
        """Analyze the following text for regional and cultural sensitivity risks.
        Give a risk score from 0 to 10,
        where 0 = completely respectful and 10 = extremely insensitive or offensive.
        Return ONLY the integer score from 0 to 10.

        Text:
        """
        f"{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except:
        score = 0

    return {"safety_score": {"cultural_risk": score}}

builder=StateGraph(AnalyzerState)

builder.add_node("toxicity_check",toxicity_node)
builder.add_node("copyright_check",copyright_node)
builder.add_node("cultural_check",cultural_sensitivity_node)

#fan-in fan-out

builder.add_edge(START,"toxicity_check")
builder.add_edge(START,"copyright_check")
builder.add_edge(START,"cultural_check")

builder.add_edge("toxicity_check",END)
builder.add_edge("copyright_check",END)
builder.add_edge("cultural_check",END)

app=builder.compile()

sample_script="People from that region are all uneducated and backward. Their culture is inferior to ours, and they should stop following their old traditions."

initial_state={
    "raw_input":sample_script,
    "safety_score":{} #Initialized an empty dictonary
}

final_state=app.invoke(initial_state)

print(final_state["safety_score"])




