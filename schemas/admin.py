from pydantic import BaseModel, Field


INT32_MAX = 1_000_000

# 관리자 신고 관련 알림
class AdminReportStatusUpdateIn(BaseModel):
    status: str
    actionResultCode: str = "NONE"
    adminMemo: str | None = None

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
    canManageUsers: bool
    canManageParties: bool
    canManageReports: bool
    canManageModeration: bool
    canApproveReceipts: bool
    canApproveSettlements: bool
    canViewLogs: bool
    canManageAdmins: bool
    lastUpdated: str
    updatedBy: str


class AdminRoleUpdateIn(BaseModel):
    canManageUsers: bool
    canManageParties: bool
    canManageReports: bool
    canManageModeration: bool
    canApproveReceipts: bool
    canApproveSettlements: bool
    canViewLogs: bool
    canManageAdmins: bool


class AdminPermissionOut(BaseModel):
    canManageUsers: bool
    canManageParties: bool
    canManageReports: bool
    canManageModeration: bool
    canApproveReceipts: bool
    canApproveSettlements: bool
    canViewLogs: bool
    canManageAdmins: bool


class AdminServiceRecordOut(BaseModel):
    id: str
    name: str
    category: str
    maxMembers: int
    monthlyPrice: int
    originalPrice: int
    logoImageKey: str | None = None
    logoImageUrl: str | None = None
    isActive: bool
    createdBy: str
    createdAt: str
    updatedAt: str
    commissionRate: float
    leaderDiscountRate: float
    referralDiscountRate: float


class AdminServiceUpdateIn(BaseModel):
    maxMembers: int = Field(ge=1)
    monthlyPrice: int = Field(ge=0, le=INT32_MAX)
    originalPrice: int = Field(ge=0, le=INT32_MAX)
    logoImageKey: str | None = None
    isActive: bool
    commissionRate: float = Field(ge=0, le=1)
    leaderDiscountRate: float = Field(ge=0, le=1)
    referralDiscountRate: float = Field(ge=0, le=1)


class AdminUserRecordOut(BaseModel):
    id: str
    name: str | None = None
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
    title: str
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
