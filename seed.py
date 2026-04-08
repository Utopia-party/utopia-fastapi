import asyncio
from core.database import AsyncSessionLocal
from core.security import hash_password
from models.user import User
from models.party import Party
# from models.party import Category, Platform
from models.notification import Notification
from sqlalchemy import select, func

import models.user  # noqa
import models.party  # noqa
import models.notification  # noqa

CATEGORIES = [
    {"category_name": "OTT"},
    {"category_name": "멤버십/음악"},
    {"category_name": "교육/도서"},
    {"category_name": "생산성"},
    {"category_name": "기타"},
]

PLATFORMS = [
    {"category_id": 1, "platform_name": "Netflix"},
    {"category_id": 1, "platform_name": "YouTube Premium"},
    {"category_id": 1, "platform_name": "Disney+"},
    {"category_id": 1, "platform_name": "Wavve"},
    {"category_id": 2, "platform_name": "Melon"},
    {"category_id": 2, "platform_name": "Spotify"},
    {"category_id": 2, "platform_name": "Apple Music"},
    {"category_id": 2, "platform_name": "쿠팡 로켓와우"},
    {"category_id": 3, "platform_name": "Class101"},
    {"category_id": 3, "platform_name": "밀리의 서재"},
    {"category_id": 3, "platform_name": "YES24"},
    {"category_id": 4, "platform_name": "ChatGPT Plus"},
    {"category_id": 4, "platform_name": "Notion"},
    {"category_id": 4, "platform_name": "Adobe Creative Cloud"},
    {"category_id": 5, "platform_name": "기타"},
]


async def seed():
    async with AsyncSessionLocal() as db:
        # cat_count = await db.scalar(select(func.count()).select_from(Category))
        # if cat_count and cat_count > 0:
        #     print("⚠️  이미 데이터가 있습니다.")
        #     return

        # for c in CATEGORIES:
        #     db.add(Category(**c))
        # await db.flush()

        # for p in PLATFORMS:
        #     db.add(Platform(**p))
        # await db.flush()

        user = User(email="test@partyup.kr", name="테스터", nickname="테스터닉", password_hash=hash_password("password123"), role="ADMIN", phone="010-0000-0000")
        db.add(user)
        await db.flush()

        # for p in [
        #     {"platform_id": 1, "title": "Netflix 프리미엄 4인 파티", "status": "RECRUITING"},
        #     {"platform_id": 5, "title": "멜론 스트리밍 같이 써요", "status": "RECRUITING"},
        #     {"platform_id": 12, "title": "ChatGPT Plus 공동 결제", "status": "RECRUITING"},
        #     {"platform_id": 9, "title": "클래스101 강의 공동구매", "status": "RECRUITING"},
        # ]:
        #     db.add(Party(id=user.id, **p))

        # db.add(Notification(id=None, type="SYSTEM", content="신뢰도 시스템 업데이트 안내입니다.", is_read=False))
        await db.commit()
        print("✅ 완료! 테스트 계정: test@partyup.kr / password123")


if __name__ == "__main__":
    asyncio.run(seed())
