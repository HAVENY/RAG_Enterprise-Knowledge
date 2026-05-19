from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class QuestionRequest(BaseModel):
    question: str = Field(..., description="用户问题")

    provider: str = Field(
        default="qwen",
        description="模型供应商：qwen/mimo/ollama/deepseek/openai",
    )

    model_level: str = Field(
        default="default",
        description="模型档位：fast/default/strong",
    )

    allow_general_answer: bool = Field(
        default=False,
        description="无知识库时是否允许通用回答",
    )
    
class QAHistoryResponse(BaseModel):
    id: int
    question: str
    answer: str
    sources: Optional[str] = None
    model_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[list] = []
    
class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 5
    
    
    