import uuid

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.database import AsyncSessionLocal
from models.party import Party, PartyMember, PartyChat, ChatReadStatus

UNIQUE_CONSTRAINT = "chat_read_status_chat_id_user_id_key"


async def _get_total_member_count(party_id: str) -> int:
    """파티 전체 인원 수 반환 (리더 포함)"""
    try:
        async with AsyncSessionLocal() as db:
            party_r = await db.execute(select(Party).where(Party.id == uuid.UUID(party_id)))
            party_obj = party_r.scalar_one_or_none()
            cnt_result = await db.execute(
                select(func.count()).select_from(PartyMember).where(
                    PartyMember.party_id == uuid.UUID(party_id),
                    PartyMember.status == "active",
                )
            )
            active_cnt = cnt_result.scalar() or 0
            if party_obj:
                ldr_check = await db.execute(
                    select(PartyMember).where(
                        PartyMember.party_id == uuid.UUID(party_id),
                        PartyMember.user_id == party_obj.leader_id,
                        PartyMember.status == "active",
                    )
                )
                ldr_in = ldr_check.scalar_one_or_none() is not None
                return active_cnt if ldr_in else active_cnt + 1
            return active_cnt
    except Exception:
        return 1


async def mark_all_read(party_id: str, user_id: str) -> None:
    """입장 시 해당 파티의 모든 메시지를 읽음 처리"""
    try:
        user_uuid = uuid.UUID(user_id)
        async with AsyncSessionLocal() as db:
            chat_ids_result = await db.execute(
                select(PartyChat.id).where(
                    PartyChat.party_id == uuid.UUID(party_id),
                    PartyChat.is_deleted == False,
                )
            )
            all_chat_ids = chat_ids_result.scalars().all()
            for cid in all_chat_ids:
                stmt = pg_insert(ChatReadStatus).values(
                    chat_id=cid,
                    user_id=user_uuid,
                ).on_conflict_do_nothing(constraint=UNIQUE_CONSTRAINT)
                await db.execute(stmt)
            await db.commit()
    except Exception as e:
        print(f"[READ ON CONNECT ERROR] {e}")


async def mark_read_for_users(chat_id: uuid.UUID, user_ids: list[uuid.UUID]) -> None:
    """메시지 전송 시 지정된 유저들을 읽음 처리"""
    try:
        async with AsyncSessionLocal() as db:
            for uid in user_ids:
                stmt = pg_insert(ChatReadStatus).values(
                    chat_id=chat_id,
                    user_id=uid,
                ).on_conflict_do_nothing(constraint=UNIQUE_CONSTRAINT)
                await db.execute(stmt)
            await db.commit()
    except Exception as e:
        print(f"[MARK READ ERROR] {e}")
