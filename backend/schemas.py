from pydantic import BaseModel


class JobDescriptionRequest(BaseModel):

    text: str


class Entity(BaseModel):

    value: str

    confidence: float


class ExtractionResponse(BaseModel):

    success: bool

    attributes: dict