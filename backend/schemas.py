from pydantic import BaseModel, Field


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