from train.experiments.scenarios.s0_full_state import SCENARIO_SPEC as S0_FULL_STATE
from train.experiments.scenarios.s1_limited_parameter import SCENARIO_SPEC as S1_LIMITED_PARAMETER
from train.experiments.scenarios.s2_payload_position import SCENARIO_SPEC as S2_PAYLOAD_POSITION
from train.experiments.scenarios.s3_clean import SCENARIO_SPEC as S3_CLEAN
from train.experiments.scenarios.s3_onboard import SCENARIO_SPEC as S3_ONBOARD
from train.experiments.scenarios.s4_robust import SCENARIO_SPEC as S4_ROBUST


SCENARIO_SPECS = {
    S0_FULL_STATE.name: S0_FULL_STATE,
    S1_LIMITED_PARAMETER.name: S1_LIMITED_PARAMETER,
    S2_PAYLOAD_POSITION.name: S2_PAYLOAD_POSITION,
    S3_CLEAN.name: S3_CLEAN,
    S3_ONBOARD.name: S3_ONBOARD,
    S4_ROBUST.name: S4_ROBUST,
}
