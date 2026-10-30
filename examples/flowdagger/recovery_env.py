"""Construct saved-state Recovery scenes without rejecting their provisional reset layout."""
import numpy as np


def make_state_restore_env(**kwargs):
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING
    from libero.libero.envs.bddl_utils import get_problem_info

    problem = get_problem_info(kwargs['bddl_file_name'])['problem_name'].lower()
    original = TASK_MAPPING[problem]

    class StateRestoreProblem(original):
        def _setup_placement_initializer(self, arena):
            super()._setup_placement_initializer(arena)
            # The evaluator immediately replaces all dynamic state with the saved
            # scene. Stacked/held objects need not fit a collision-free table reset.
            # Keep fixture definitions/regions and goal predicates. MuJoCo
            # collision physics remains enabled after saved-state restoration.
            for initializer in (self.placement_initializer, self.conditional_placement_initializer,
                                self.conditional_placement_on_objects_initializer):
                for sampler in initializer.samplers.values():
                    sampler.ensure_valid_placement = False

    # BDDLBaseDomain checks the Python class name against the BDDL problem name.
    StateRestoreProblem.__name__ = original.__name__

    class StateRestoreEnv(OffScreenRenderEnv):
        def set_init_state(self, state):
            observation = super().set_init_state(state)
            actual = self.sim.get_state().flatten()
            if not np.allclose(actual, state, rtol=0, atol=1e-9):
                raise ValueError('Saved Recovery state was not restored exactly before settling')
            return observation

    # The instance retains its subclass for subsequent hard resets. Registration
    # is restored immediately; other environments keep their original behavior.
    TASK_MAPPING[problem] = StateRestoreProblem
    try:
        return StateRestoreEnv(**kwargs)
    finally:
        TASK_MAPPING[problem] = original
