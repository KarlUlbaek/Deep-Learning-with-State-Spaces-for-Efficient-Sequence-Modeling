__version__ = "1.2.0.post1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import S6MambaModule
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
