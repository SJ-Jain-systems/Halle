from src.solr_client import doi_to_xml_path, psychology_subfields_from_paths


def test_subfields_from_paths_extracts_segment_after_psychology():
    paths = [
        "/Biology and life sciences/Psychology/Cognitive psychology/Decision making",
        "/Social sciences/Psychology/Cognitive psychology/Decision making",
        "/Biology and life sciences/Psychology/Behavior/Animal behavior",
    ]
    assert psychology_subfields_from_paths(paths) == ["Cognitive psychology", "Behavior"]


def test_subfields_from_paths_psychology_leaf():
    assert psychology_subfields_from_paths(["/Social sciences/Psychology"]) == ["Psychology"]


def test_subfields_from_paths_ignores_paths_without_psychology_node():
    # "Cognitive psychology" nested under Neuroscience is not a Psychology-node child.
    paths = ["/Biology and life sciences/Neuroscience/Cognitive science/Cognitive psychology/Language"]
    assert psychology_subfields_from_paths(paths) == []


def test_subfields_from_paths_empty_when_no_psychology():
    paths = ["/Medicine and health sciences/Oncology", "/Physical sciences/Physics"]
    assert psychology_subfields_from_paths(paths) == []


def test_doi_to_xml_path():
    assert doi_to_xml_path("10.1371/journal.pone.0116314", "/corpus") == "/corpus/journal.pone.0116314.xml"
