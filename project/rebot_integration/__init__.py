"""Import to register the B601-DM model, fixed-base wrapper, and gripper."""

from .rebot_b601_dm_gripper import ReBotB601DMGripper
from .rebot_b601_dm import ReBotB601DM

__all__ = ["ReBotB601DM", "ReBotB601DMGripper"]
