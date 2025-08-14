from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone


class Paper(BaseModel):
    title: str = Field(..., max_length=300)
    author: str = Field(..., max_length=100)
    published_year: int = Field(..., ge=1500, le=datetime.now().year)

class Claim(BaseModel):
    content: str = Field(..., max_length=1000)

class ListClaims(BaseModel):
    listofClaims: list[Claim]

class Logic(BaseModel):
    logic: Literal[
        "support",
        "reject",
        "mention",
    ] = Field(..., description="Relation type from set of categories")
