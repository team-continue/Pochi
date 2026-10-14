"""Independent LP-ACRL walking experiment; see arXiv:2601.17428."""

from pochi_rl.lp_acrl.curriculum import LPACRLCurriculum, StageUpdate
from pochi_rl.lp_acrl.metrics import epte_sp, percentage_tracking_error
from pochi_rl.lp_acrl.spec import (
  FLAT_TASK_CATALOG,
  LP_ACRL_SPEC,
  ROUGH_TASK_CATALOG,
  LPACRLSpec,
  TaskCatalog,
)

__all__ = [
  "FLAT_TASK_CATALOG",
  "LP_ACRL_SPEC",
  "ROUGH_TASK_CATALOG",
  "LPACRLCurriculum",
  "LPACRLSpec",
  "StageUpdate",
  "TaskCatalog",
  "epte_sp",
  "percentage_tracking_error",
]
