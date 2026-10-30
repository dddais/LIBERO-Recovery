import numpy as np

from examples.flowdagger.protocol import build_plan, perturb_xy, policy_observation, summarize
from examples.flowdagger.evaluate import rollout, settle


def test_perturbation_updates_qpos_not_time_or_neighbor_coordinates():
    state = np.arange(20, dtype=float)
    perturbed, deltas = perturb_xy(state, [("bowl", 3, 4)], 0.02, np.random.default_rng(7))
    difference = perturbed - state
    np.testing.assert_array_equal(np.flatnonzero(difference), [4, 5])
    np.testing.assert_allclose(difference[[4, 5]], deltas["bowl"])
    np.testing.assert_array_equal(state, np.arange(20))


def test_settle_returns_final_observation():
    class Env:
        count = 0
        def step(self, action):
            assert action[-1] == 1
            self.count += 1
            return {"step": self.count}, 0, False, {}
    assert settle(Env(), {"step": 0}, "closed", 20) == {"step": 20}


def raw_observation():
    return {"agentview_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            "robot0_eye_in_hand_image": np.full((2, 3, 3), 255, np.uint8),
            "robot0_eef_pos": [1, 2, 3], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [0.01, -0.01]}


def test_environment_observation_has_native_state_and_one_image_rotation():
    task = {"task_key": "suite/task", "official_task_language": "pick up"}
    obs = raw_observation()
    request = policy_observation(obs, task, 9)
    np.testing.assert_array_equal(request["observation/image"], obs["agentview_image"][::-1, ::-1])
    np.testing.assert_allclose(request["observation/state"], [1, 2, 3, 0, 0, 0, 0.01, -0.01])
    assert request["prompt"] == "pick up"


def test_rollout_replans_and_does_not_recode_gripper():
    class Env:
        actions = []
        def step(self, action):
            self.actions.append(action)
            return raw_observation(), 0, False, {}
        def check_success(self):
            return len(self.actions) == 5
    class Client:
        requests = []
        def infer(self, request):
            self.requests.append(request)
            return {"actions": np.tile([0, 0, 0, 0, 0, 0, -1], (10, 1))}
    env, client = Env(), Client()
    task = {"task_key": "suite/task", "official_task_language": "pick up"}
    success, steps = rollout(env, raw_observation(), task, client, {"seed": 42}, 2, 20)
    assert success and steps == 5 and len(client.requests) == 3
    assert all(a[-1] == -1 for a in env.actions)


def test_plan_is_reproducible_and_source_task_mapping_is_explicit(tmp_path):
    directory = tmp_path / "suite" / "task"
    directory.mkdir(parents=True)
    for i in range(4):
        (directory / f"step_{i}.bddl").touch()
        np.save(directory / f"step_{i}_sim_state.npy", np.zeros(20))
    tasks = {"suite/task": {"task_index": 9, "task_key": "suite/task", "suite": "suite", "task_id": "task"}}
    arguments = (tasks, "recovery", tmp_path, 42, 2, 25, 1, 1, "*.bddl")
    first = build_plan(*arguments)
    assert first == build_plan(*arguments)
    assert len(first) == 4 and {r["gripper"] for r in first} == {"closed", "open"}
    assert all(r["task_index"] == 9 and r["task_key"] == "suite/task" for r in first)


def test_summary_keeps_failed_initializations_in_planned_denominator():
    results = [dict(task_key="task", gripper="open", status=status, success=success)
               for status, success in [("completed", True), ("completed", False), ("reset_error", False), ("initial_success", False)]]
    counts = summarize(results)["overall"]
    assert counts["success_rate_planned"] == 0.25
    assert counts["success_rate_valid"] == 0.5
    assert counts["reset_errors"] == 1 and counts["initially_successful"] == 1
