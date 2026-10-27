"""StormCrews: a simulator of distribution restoration after storms, on real storms.

    from stormcrews import Simulator, rules
    result = Simulator("new_england").run(rules.CurrentPractice(), storm=10, seed=1)
"""
from . import rules
from .core.data import Region, region
from .core.engine import Decision
from .damage import DamageModel, model as damage_model
from .experiment import Experiment, paired_comparison, run_experiment
from .metrics import restoration_curve, storm_metrics
from .scenario import Scenario, shipped
from .simulator import Result, Session, Simulator, State
from .synthetic import gust_field, synthetic_storm
from .whatif import WhatIf

__version__ = "0.3.0"

__all__ = ["Simulator", "Session", "State", "Result", "Decision", "Scenario", "shipped", "run_experiment",
           "Experiment", "paired_comparison", "rules", "Region", "region", "storm_metrics", "restoration_curve",
           "WhatIf", "DamageModel", "damage_model", "gust_field", "synthetic_storm", "__version__"]
