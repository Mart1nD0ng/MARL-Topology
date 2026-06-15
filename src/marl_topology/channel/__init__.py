"""Stage 3 channel simulation records and deterministic v1 model."""

from .fixtures import (
    ChannelModelFixture,
    get_channel_model_fixture,
    iter_channel_model_fixtures,
)
from .model import (
    ActiveTransmission,
    ChannelModelConfig,
    ChannelRecord,
    PATH_LOSS_MODEL_FSPL,
    PATH_LOSS_MODEL_UMI,
    PATH_LOSS_MODEL_V2X,
    STAGE3_CHANNEL_REGIME_ID,
    dbm_to_mw,
    evaluate_channel,
    free_space_path_loss_db,
    mw_to_dbm,
    noise_power_dbm,
    umi_street_canyon_path_loss_db,
    v2v_37885_path_loss_db,
)

__all__ = [
    "ActiveTransmission",
    "ChannelModelConfig",
    "ChannelModelFixture",
    "ChannelRecord",
    "PATH_LOSS_MODEL_FSPL",
    "PATH_LOSS_MODEL_UMI",
    "PATH_LOSS_MODEL_V2X",
    "STAGE3_CHANNEL_REGIME_ID",
    "dbm_to_mw",
    "evaluate_channel",
    "free_space_path_loss_db",
    "get_channel_model_fixture",
    "iter_channel_model_fixtures",
    "mw_to_dbm",
    "noise_power_dbm",
    "umi_street_canyon_path_loss_db",
    "v2v_37885_path_loss_db",
]
