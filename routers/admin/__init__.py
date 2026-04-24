from fastapi import APIRouter

from .dashboard import router as dashboard_router
from .roles import router as roles_router
from .users import router as users_router
from .parties import router as parties_router
from .reports import router as reports_router
from .receipts import router as receipts_router
from .settlements import router as settlements_router
from .logs import router as logs_router
from .moderation import router as moderation_router
from .captcha_admin import router as captcha_admin_router
from .payments_admin import router as payments_admin_router

router = APIRouter()

router.include_router(dashboard_router)
router.include_router(roles_router)
router.include_router(users_router)
router.include_router(parties_router)
router.include_router(reports_router)
router.include_router(receipts_router)
router.include_router(settlements_router)
router.include_router(logs_router)
router.include_router(moderation_router)
router.include_router(captcha_admin_router)
router.include_router(payments_admin_router)