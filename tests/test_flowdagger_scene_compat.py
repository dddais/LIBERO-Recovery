from pathlib import Path
from examples.flowdagger.scene_compat import prepare_bddl, repair_text


def test_grouping_keeps_all_instances_and_original_order():
    text = '(define (problem x) (:objects\n a_1 - bowl\n a_2 - bowl\n p_1 - plate\n) (:goal (And (on a_1 p_1))))'
    result, changes = repair_text(text)
    assert 'a_1 a_2 - bowl' in result
    assert 'p_1 - plate' in result
    assert '(:goal (And (on a_1 p_1)))' in result
    assert changes == ['merge_repeated_objects_categories']
    assert repair_text(result) == (result, [])


def test_floor_alias_is_scoped_and_copies_leave_source_untouched(tmp_path):
    text = '(define (problem libero_floor_manipulation) (:fixtures\n floor - floor\n) (:regions (r (:target main_table))) (:init (On cheese main_table_r)))'
    source = tmp_path / 'scene.bddl'
    source.write_text(text)
    runtime, report = prepare_bddl(source, tmp_path / 'derived')
    assert source.read_text() == text
    assert 'main_table' not in Path(runtime).read_text()
    assert '(On cheese floor_r)' in Path(runtime).read_text()
    assert report['changes'] == ['floor_workspace_alias']
    assert prepare_bddl(source, tmp_path / 'derived') == (runtime, report)
    other = text.replace('libero_floor_manipulation', 'another_problem')
    assert repair_text(other) == (other, [])


def test_real_exported_recovery_scenes_preserve_goal_and_fix_declarations():
    root = Path(__file__).resolve().parents[1]
    bowl = root / 'benchmark_failure/scenes/libero_90/KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate/episode_20261009_173231_2599_000/step_0120.bddl'
    original = bowl.read_text()
    repaired, changes = repair_text(original)
    assert 'akita_black_bowl_1 akita_black_bowl_2 akita_black_bowl_3 - akita_black_bowl' in repaired
    assert repaired[repaired.index('(:goal'):] == original[original.index('(:goal'):]
    assert changes == ['merge_repeated_objects_categories']
