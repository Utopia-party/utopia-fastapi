import uuid
from pydantic import BaseModel
from typing import Optional


class ServiceOut(BaseModel):
    id: uuid.UUID
    name: str
    category: str
    max_members: int
    monthly_price: int
    logo_image_key: Optional[str]
    logo_image_url: Optional[str] = None
    is_active: bool

    model_config = {"from_attributes": True}


# 프론트 /categories 호환용
class CategoryOut(BaseModel):
    category_id: uuid.UUID
    category_name: str

    model_config = {"from_attributes": True}
