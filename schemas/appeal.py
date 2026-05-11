from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AppealCreateIn(BaseModel):
    user_id: Optional[str] = None    
    ban_type: str
    ban_reference_id: Optional[str] = None
    reason: str


class AppealOut(BaseModel):
    id: str
    user_id: str
    ban_type: str
    ban_reference_id: Optional[str]
    reason: str
    status: str
    admin_memo: Optional[str]
    created_at: str

    class Config:
        from_attributes = True

class AdminAppealOut(BaseModel):
    id: str
    user_id: str
    user_nickname: str
    user_email: str
    ban_type: str
    ban_reference_id: Optional[str]
    reason: str
    status: str
    admin_memo: Optional[str]
    reviewed_by_nickname: Optional[str]
    reviewed_at: Optional[str]
    created_at: str
    ban_detail: Optional[str]       
    ban_score_change: Optional[float]  
    ban_created_at: Optional[str]
    ip_address: Optional[str]


class AdminAppealReviewIn(BaseModel):
    status: str    
    admin_memo: Optional[str] = None
