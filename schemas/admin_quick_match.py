from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepTimingsOut(BaseModel):
    validationMs: int = 0
    profileEmbeddingMs: int = 0
    hardFilterMs: int = 0
    ruleScoringMs: int = 0
    vectorScoringMs: int = 0
    joinPartyMs: int = 0


class AdminQuickMatchCandidateOut(BaseModel):
    candidateId: str
    partyId: str
    partyName: str
    rank: int | None = None
    status: str
    ruleScore: float
    vectorScore: float
    finalScore: float
    filterReasons: dict[str, Any] = Field(default_factory=dict)


class AdminQuickMatchProfileSnapshotOut(BaseModel):
    trustScore: float = 0
    preferredConditions: dict[str, Any] = Field(default_factory=dict)
    activitySummary: dict[str, Any] = Field(default_factory=dict)
    paymentSummary: dict[str, Any] = Field(default_factory=dict)
    riskSummary: dict[str, Any] = Field(default_factory=dict)


class AdminQuickMatchRequestOut(BaseModel):
    requestId: str
    requestedAt: str
    userId: str
    userNickname: str
    serviceName: str
    status: str
    matchedPartyId: str | None = None
    matchedPartyName: str | None = None
    totalMatchSeconds: float | None = None
    retryCount: int
    failReason: str | None = None
    stepTimings: StepTimingsOut
    aiProfileSnapshot: AdminQuickMatchProfileSnapshotOut
    candidates: list[AdminQuickMatchCandidateOut] = Field(default_factory=list)


class AdminQuickMatchSummaryOut(BaseModel):
    total: int
    todayTotal: int
    matched: int
    successRate: float
    avgSeconds: float
    stepAvg: StepTimingsOut


class AdminQuickMatchListOut(BaseModel):
    summary: AdminQuickMatchSummaryOut
    rows: list[AdminQuickMatchRequestOut]
    total: int
    page: int
    pageSize: int


class AdminQuickMatchPolicyOut(BaseModel):
    quickMatchEnabled: bool = True
    topN: int = 5
    maxCandidates: int = 30
    minMatchScore: float = 0.7
    vectorWeight: float = 0.5
    trustWeight: float = 0.4
    capacityWeight: float = 0.3
    durationWeight: float = 0.3
    joinPartyLockTtlSeconds: int = 30
    maxRetry: int = 3


class AdminQuickMatchPolicyResponse(BaseModel):
    policy: AdminQuickMatchPolicyOut


class AdminQuickMatchActionResponse(BaseModel):
    success: bool
    message: str | None = None