from typing import Optional
from pydantic import BaseModel


class SaasKeyContext(BaseModel):
    api_key_id: str
    client_name: str
    api_key: str
    secret_key: str
    service_type: str   
    plan: str
    allowed_domains: Optional[list[str]] = None
    monthly_limit: int
    current_month_usage: int
    matched_domain: Optional[str] = None
