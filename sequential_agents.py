# 1) Project sequential workflow
# So I am a content creator and I write scripts but the raw script is not as good as I want,
# so I will create a multi-stage content pipeline to generate a good script.

# Ideation:-
# Raw Data → Editor Node → Scriptwriter Node → Hinglish Node → O/P

# [start]
# Editor Node: Cleans up grammar and tone
# Scriptwriter Node: Formats it into an engaging script
# Hinglish Node: Converts it into Hinglish format
# O/P [END]

import os
from typing import TypedDict

#lets create the state first
class PipelineState(TypedDict):
    raw_input:str
    edited_text:str
    script_text:str
    final_output:str

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(model="openai/gpt-oss-120b",temperature=1.5)

def editor_node(state:PipelineState)->dict:
    """" Stage 1: Cleans up grammar,removes typos, and refines the tone"""

    print("\n --[Stage 1 ] is Editor Node--")

    prompt=("""You are a professional English editor. 
            Your task is to improve the given text by:
            Correcting grammar and spelling mistakes.
            Removing typos and obvious writing errors.
            Improving sentence structure and readability.
            Refining the tone to make it clear, natural, and professional.
            Preserving the original meaning, intent, and important information.
            Do not add new information or change the author's message.
            Keep the output concise and natural.
            Return only the improved version of the text. Do not explain the changes."""
            f"Text :/n{state['raw_input']}"
            )

    response=llm.invoke(prompt)

    return {"edited_text":response.content.strip()} #response pipelinestate ke edited_text me chala jayega

def scriptwriter_node(state:PipelineState)->dict:
    """Stage2:Format the clean text into an engaging video script style """

    print("\n --[Stage 2 ] is Executing Scriptwriter Node--")

    prompt=(""" Take the cleaned and edited text provided below and transform it into an engaging YouTube video script.
                Your goals:
                Start with a strong HOOK that immediately grabs the viewer's attention.
                Make the script conversational, natural, and energetic—as if a charismatic YouTuber is speaking directly to the audience.
                Convert formal or plain sentences into engaging spoken-language sentences.
                Maintain the original meaning, facts, and key information. Do not invent facts or add unsupported information.
                Improve the flow by creating smooth transitions between ideas.
                Use curiosity, storytelling, and audience engagement where appropriate.
                Add rhetorical questions or short audience-directed lines when they naturally fit.
                Break long paragraphs into short, easy-to-speak lines.
                Story must has atleast 2000 lines"""
            f"Edited Text:\n{state['edited_text']}")

    response=llm.invoke(prompt)

    return {"script_text":response.content.strip()}


def translator_node(state:PipelineState) ->dict:
    """ Stage 3 :Translator the script into natural flowing Hinglish"""
    print("\n--- [Stage 3] executing HINGLISH translator Node---")

    prompt=("""You are an expert Hinglish translator and YouTube script localization specialist.
        Translate the given English script into natural, fluent, conversational Hinglish that
        sounds like it was originally written by a native Indian YouTube content creator.

        Follow these rules:
        Do NOT translate word-for-word or literally.
        Preserve the original meaning, facts, context, emotion, and intent.
        Use a natural mix of Hindi(but written in english sound hindi) and commonly used English words, the way Indians normally speak in YouTube videos.
        Keep technical terms, proper nouns, names, brand names, scientific terms, and commonly understood English words in English when appropriate.
        Avoid overly formal, textbook-style Hindi and difficult Hindi vocabulary."""
        f"Script :\n{state['script_text']}"
    )

    response=llm.invoke(prompt)

    return {"final_output":response.content.strip()}

#now your state and nodes are ready and now it is time to create the graph
#and for creating the graph you have to connect these nod3s and for that you have
#to use the edges
#edges are very important to create the workflows

from langgraph.graph import StateGraph,START,END

#create garph
graph=StateGraph(PipelineState)

#add the node in our graph
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

#add edges (sequential conection one after another)
graph.add_edge(START,"editor")
graph.add_edge('editor',"scriptwriter")
graph.add_edge("scriptwriter","translator")
graph.add_edge("translator",END)


#compile the graph

app=graph.compile()

#raw_input pipelinestate ka key hai 
result=app.invoke({
    "raw_input":"AI agents are the future of tech. They can think,plan and act on their own .Langraphs heps you to build thses agents with proper control and memory"

})

#output
print("--YOur Result---")
print(result["final_output"])






