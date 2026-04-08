from pydantic import BaseModel


class DashboardMetricOut(BaseModel):
    id: str
    label: str
    value: str
    helper: str


class DashboardSummaryRowOut(BaseModel):
    label: str
    value: str


class AdminDashboardOut(BaseModel):
    metrics: list[DashboardMetricOut]
    member_stats: list[DashboardSummaryRowOut]
    sales_stats: list[DashboardSummaryRowOut]
    today_summary: str


class AdminRoleRecordOut(BaseModel):
    id: str
    userId: str
    adminId: str
    role: str
    scope: str
    lastUpdated: str
    updatedBy: str


class AdminRoleUpdateIn(BaseModel):
    role: str


class AdminUserRecordOut(BaseModel):
    id: str
    nickname: str
    status: str
    reportCount: int
    partyCount: int
    trustScore: int
    lastActive: str


class AdminUserDetailOut(BaseModel):
    id: str
    email: str
    nickname: str
    name: str | None = None
    phone: str | None = None
    role: str
    status: str
    trustScore: int
    reportCount: int
    partyCount: int
    createdAt: str | None = None
    lastActive: str | None = None


class AdminUserStatusUpdateIn(BaseModel):
    status: str
    reason: str | None = None


class AdminPartyRecordOut(BaseModel):
    id: str
    service: str
    leaderId: str
    memberCount: int
    status: str
    reportCount: int
    monthlyAmount: int
    lastPayment: str


class AdminPartyActionIn(BaseModel):
    reason: str | None = None


class ReportRecordOut(BaseModel):
    id: str
    type: str
    target: str
    reason: str
    status: str
    content: str
    createdAt: str


class AdminStatusUpdateIn(BaseModel):
    status: str


class ReceiptRecordOut(BaseModel):
    id: str
    userId: str
    partyId: str
    ocrAmount: int
    status: str
    createdAt: str


class SettlementRecordOut(BaseModel):
    id: str
    partyId: str
    leaderId: str
    totalAmount: int
    memberCount: int
    billingMonth: str
    status: str
    createdAt: str


class SystemLogRecordOut(BaseModel):
    id: str
    timestamp: str
    type: str
    message: str
    actor: str
