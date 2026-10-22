#Now we are creating a graph
#The first thing we will create will be states

import os

from typing import TypedDict

#1st way of creating state TypedDict
class State(TypedDict):
    topic:str
    summary:str
    score:str

#2 way of creating is Pydantic
#It is good at validation and typechecking during runtime

from pydantic import BaseModel,field_validator

class State(BaseModel):
    topic:str
    summary:str
    score:int

    @field_validator("score")
    @classmethod
    def score_positive(cls,v):
        if v<0:
            raise ValueError("Score must be positive")


#3 way python-Dataclasses
#standard python dataclass but it used rarely

from dataclasses import dataclass,field

@dataclass
class State:
    topic:str=""
    summary:str=""
    messages:list=field(default_factory=list)

#4th langragph

from langgraph.graph import MessagesState
class State(MessagesState):
  user_name:str
  language:str




