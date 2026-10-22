import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

#tool
search_tool=TavilySearch(max_results=3)

tools=[search_tool]

#writer
writer_llm=ChatGroq(model="openai/gpt-oss-120b",temperature=1.5)
writer_llm_with_tool=writer_llm.bind_tools(tools)

#reviewer
reviewer_llm=ChatGroq(model="qwen/qwen3.8-27b",temperature=0.3)

#state
class State(TypedDict):
    topic:str
    messages:Annotated[list,add_messages]
    draft:str
    review_feedback:str
    is_approved:bool
    attempt:int


#node

#writer_node
writer_system_prompt = (
    "You are an expert LinkedIn content writer specializing in AI, ML, "
    "software development, technology, engineering, and career topics. "
    "Create an engaging, accurate, human-sounding LinkedIn post about "
    "the user's topic. "
    "Follow this structure: "
    "1. Hook: 1-3 attention-grabbing lines. "
    "2. Context: Why the topic matters and where it is used. "
    "3. Explanation: Define it, explain how it works, key terms, "
    "practical example, and common mistakes. "
    "4. Real-world example. "
    "5. 3-6 key takeaways. "
    "6. Short learning insight. "
    "7. End with one engaging question. "
    "Style: Simple, conversational, professional English. "
    "Use short, mobile-friendly paragraphs and emojis sparingly. "
    "Avoid jargon, clichés, buzzwords, repetition, fake statistics, "
    "unsupported claims, and excessive promotion. "

    "Length: 500-900 words. "
    "End with 5-8 relevant hashtags. "

    "Return ONLY the final LinkedIn post. "
    "Do not explain your reasoning or add introductory text."
)

def writer_node(state:State)-> dict:
    """ wites (or review ) the linkedin post.Can call tavily to search first"""
    attempt=state.get('attempt',0)+1
    topic=state['topic']
    prev_feedback=state['review_feedback']


    if attempt==1: 
        user_message=(
            f"Write a linkedin post on this topic {topic}"
            f"if you need current info search the web first"
        )
    else:
        user_message=(
            f"Your previous draft on '{topic} was rejected"
            f"Here is the revier feedback \n\n {prev_feedback}\n\n"
            f"Write a new , improved draft that fixes every issue mentioned"
            f"Do not reapeat the same mistake"
        )
    messages=[("system",writer_system_prompt),("human",user_message)]
    response=writer_llm_with_tool.invoke(messages)


    return {
        "messages":[("human",user_message),response],
        "attempt":attempt
    }


tool_node=ToolNode(tools)

#draft node
def extract_draft_node(state:State)->dict:
    """After the writer finishes tool class, pull the final text out as the draft ."""
    last_msg=state['messages'][-1]
    draft=last_msg.content
    print(f"\n\n Generated post \n {draft}")

    return {"draft":draft}

#reviewer node

reviewer_system_prompt = (
    "You are a strict LinkedIn content reviewer."

    "You judge whether a LinkedIn post is publication-ready. "
    "Evaluate the post objectively against the following criteria:\n"

    "1. HOOK:\n"
    "- Does the opening grab attention within the first 1-3 lines?\n"
    "- Is it relevant to the topic?\n"
    "- Does it create curiosity without using misleading clickbait?\n"

    "2. CONTENT QUALITY:\n"
    "- Is the information clear, useful, and relevant?\n"
    "- Does the post provide genuine value to the reader?\n"
    "- Is the topic explained correctly and logically?\n"
    "- Are examples practical and understandable?\n"

    "3. TECHNICAL ACCURACY:\n"
    "- Check technical concepts, terminology, code, and claims.\n"
    "- Identify incorrect or misleading information.\n"
    "- Flag unsupported statistics, benchmarks, or factual claims.\n"
    "- Do not approve content containing obvious technical errors.\n"

    "4. STRUCTURE:\n"
    "- Is the post easy to follow?\n"
    "- Does it have a logical flow from hook to conclusion?\n"
    "- Are paragraphs short and mobile-friendly?\n"
    "- Are bullet points used appropriately?\n"

    "5. READABILITY:\n"
    "- Is the language simple and understandable?\n"
    "- Avoid unnecessary jargon and complicated sentences.\n"
    "- Check grammar, spelling, punctuation, and sentence clarity.\n"

    "6. ENGAGEMENT:\n"
    "- Does the post encourage meaningful discussion?\n"
    "- Is the final question relevant and natural?\n"
    "- Avoid forced engagement bait such as 'Agree?' or 'Comment YES'.\n"

    "7. HUMAN WRITING:\n"
    "- Does the post sound natural and human?\n"
    "- Flag generic AI-sounding phrases, clichés, and repetitive wording.\n"
    "- Avoid excessive motivational language and corporate buzzwords.\n"

    "8. LINKEDIN STYLE:\n"
    "- Is the tone professional but conversational?\n"
    "- Is the formatting suitable for LinkedIn?\n"
    "- Are emojis used sparingly and appropriately?\n"
    "- Are hashtags relevant and limited to approximately 5-8?\n"

    "9. ORIGINALITY AND VALUE:\n"
    "- Does the post provide a useful perspective or explanation?\n"
    "- Avoid generic content that could apply to any topic.\n"
    "- Identify unnecessary repetition or filler content.\n"

    "10. FINAL DECISION:\n"
    "Classify the post as one of:\n"
    "- PASS: Publication-ready with no significant issues.\n"
    "- REVISE: Useful content but requires specific improvements.\n"
    "- REJECT: Contains major factual, structural, or quality problems.\n"

    "When reviewing, do not rewrite the entire post. "
    "Identify the specific problems and explain how they should be fixed.\n"

    "Return the review in this format:\n"
    "Decision: PASS / REVISE / REJECT\n"
    "Score: X/10\n"
    "Strengths:\n"
    "- ...\n"
    "- ...\n"
    "Issues:\n"
    "- ...\n"
    "- ...\n"
    "Required Improvements:\n"
    "- ...\n"
    "- ...\n"
)


def reviwer_node(state:State) ->dict:
    """Review the draft and decides approve or reject with feedback."""
    draft=state['draft']

    prompt=(
        f"Review this linkedIn post draft :\n"
        f"{draft}"
        f"Give your reviews"
    )

    response=reviewer_llm.invoke(
        [("system",reviewer_system_prompt),("human",prompt)]
    )

    review_text=response.content.strip()

    is_approved = "DECISION: PASS" in review_text.upper()
    feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"Verdict: {verdict}")
    print(f"Feedback: {feedback}")

    return {
            "review_feedback": feedback,
            "is_approved": is_approved
        }

#router

def should_use_tool(state:State):
    last_message=state['messages'][-1]
    # if getattr[last_message,'tool_calls',None]:
    if getattr(last_message,'tool_calls',None):
        return "tools"
    return "extract_draft"

def should_stop_looping(state:State):
    if state['is_approved']:
        print("\n\n---POST HAS BEEN APPROVED-----")
        return END
    if state['attempt']>=3:
        print("Reached max attempt")
        return END
    return "writer"

#graph
graph=StateGraph(State)

graph.add_node("writer",writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft",extract_draft_node)
graph.add_node("reviewer",reviwer_node)

#edge

graph.add_edge(START,"writer")
graph.add_conditional_edges("writer",should_use_tool)

graph.add_edge("tools","reviewer")

graph.add_edge("extract_draft","reviewer")

graph.add_conditional_edges("reviewer",should_stop_looping)

app=graph.compile()

print("="*50)
print("Welcome to the LinkedIn Post Generator")
print("\nThis tool will draft a linkedIn post for you,review it")
print("itself ,and iterate untill it's a publish-ready")
print("="*50)

topic=input("\nWhat topic do you want a linkedIn post about ?\n").strip()

if not topic:
    print("\nExiting generation...\n")
else:
    print("\nStarting generation")

    initial_state={
           "topic":topic,
            "messages":[],
           "draft":"",
            "review_feedback":"",
            "is_approved":False,
            "attempt":0
            }
    final_state=app.invoke(initial_state)

    print("\n"+"="*55)
    print("FINAL LINKEDIN POST")
    print("="*50)
    print(f"Total attempt :{final_state['attempt']}")
    print(f"Approved :{final_state['is_approved']}")


