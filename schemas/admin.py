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
    delta: str | None = None
    trend: str | None = None


class DashboardSummaryRowOut(BaseModel):
    label: str
    value: str


class DashboardSeriesPointOut(BaseModel):
    label: str
    current: int
    comparison: int


class DashboardChartOut(BaseModel):
    id: str
    label: str
    description: str
    unit: str
    points: list[DashboardSeriesPointOut]


class DashboardRecentActivityOut(BaseModel):
    timestamp: str
    title: str
    description: str


class AdminUserAccessLogOut(BaseModel):
    id: str
    ipAddress: str | None = None
    userAgent: str | None = None
    createdAt: str
    isActive: bool = False


class AdminUserTrustHistoryOut(BaseModel):
    id: str
    title: str
    detail: str | None = None
    scoreChange: float
    trustScoreAfter: float
    createdAt: str
    changedBy: str


class AdminModerationHistoryOut(BaseModel):
    id: str
    actionType: str
    reason: str | None = None
    trustScoreChange: float | None = None
    durationMinutes: int | None = None
    createdAt: str
    createdBy: str


class AdminUserStatusLogOut(BaseModel):
    id: str
    toStatus: str
    changedBy: str
    reason: str | None = None
    trigger: str
    createdAt: str


class AdminDashboardOut(BaseModel):
    metrics: list[DashboardMetricOut]
    member_stats: list[DashboardSummaryRowOut]
    sales_stats: list[DashboardSummaryRowOut]
    today_summary: str
    period_label: str
    comparison_label: str
    compare_mode: str
    range_start: str
    range_end: str
    chart_points: list[DashboardSeriesPointOut]
    chart_groups: list[DashboardChartOut]
    recent_activities: list[DashboardRecentActivityOut]


class AdminRoleRecordOut(BaseModel):
    id: str
    userId: str
    adminId: str
    canViewDashboard: bool
    canManageUsers: bool
    canManageServices: bool
    canManageParties: bool
    canManageQuickMatch: bool
    canManageReports: bool
    canManageChatModeration: bool
    canManageCaptcha: bool
    canApproveSettlements: bool
    canManagePayments: bool
    canManageHandOcr: bool
    canViewLogs: bool
    canViewCloudMonitoring: bool
    canManageAdmins: bool
    lastUpdated: str
    updatedBy: str


class AdminRoleUpdateIn(BaseModel):
    canViewDashboard: bool
    canManageUsers: bool
    canManageServices: bool
    canManageParties: bool
    canManageQuickMatch: bool
    canManageReports: bool
    canManageChatModeration: bool
    canManageCaptcha: bool
    canApproveSettlements: bool
    canManagePayments: bool
    canManageHandOcr: bool
    canViewLogs: bool
    canViewCloudMonitoring: bool
    canManageAdmins: bool


class AdminPermissionOut(BaseModel):
    canViewDashboard: bool
    canManageUsers: bool
    canManageServices: bool
    canManageParties: bool
    canManageQuickMatch: bool
    canManageReports: bool
    canManageChatModeration: bool
    canManageCaptcha: bool
    canApproveSettlements: bool
    canManagePayments: bool
    canManageHandOcr: bool
    canViewLogs: bool
    canViewCloudMonitoring: bool
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
    createdAt: str
    status: str
    reportCount: int
    partyCount: int
    trustScore: float
    lastActive: str


class AdminUserDetailOut(BaseModel):
    id: str
    email: str
    nickname: str
    name: str | None = None
    phone: str | None = None
    role: str
    status: str
    trustScore: float
    reportCount: int
    partyCount: int
    createdAt: str | None = None
    lastActive: str | None = None
    bannedUntil: str | None = None
    recentLoginIp: str | None = None
    recentLoginUserAgent: str | None = None
    recentLoginAt: str | None = None
    trustHistories: list[AdminUserTrustHistoryOut] = []
    accessLogs: list[AdminUserAccessLogOut] = []
    moderationHistories: list[AdminModerationHistoryOut] = []


class AdminUserStatusUpdateIn(BaseModel):
    status: str
    reason: str | None = None


class AdminUserTrustScoreUpdateIn(BaseModel):
    trustScore: float
    reason: str | None = None


class AdminPartyRecordOut(BaseModel):
    id: str
    title: str
    createdAt: str
    service: str
    category: str
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
    partyName: str
    leaderId: str
    leaderName: str
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

class AdminPartyMemberOut(BaseModel):
    memberId: str        
    userId: str
    nickname: str
    name: str | None = None
    role: str           
    status: str          
    trustScore: float
    joinedAt: str
    leftAt: str | None = None

class AdminPartyMemberKickIn(BaseModel):
    reason: str | None = None

class AdminPartyMemberRoleIn(BaseModel):
    role: str            

class ChatModerationLogOut(BaseModel):
    id: str
    partyId: str
    partyTitle: str
    senderId: str
    senderNickname: str
    message: str
    flagReason: str | None = None
    flagConfidence: float | None = None
    flagStage: int | None = None
    moderationStatus: str | None = None
    isDeleted: bool
    createdAt: str
    warnCount: int | None = None

class ChatModerationStatsOut(BaseModel):
    totalFlagged: int
    blocked: int
    warned: int
    falsePositive: int
    pending: int
    detectionRate: float   

class UserStatusLogOut(BaseModel):
    id: str
    toStatus: str          
    changedBy: str         
    reason: str | None = None
    trigger: str           
    createdAt: str
