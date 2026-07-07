from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from tiz.helpers import (
    _format_exc,
    ask_confirmations,
    can_parallelize,
    chat,
    exec_cmd,
    get_credits,
    parse_manifest,
    run,
)
from tiz.manifest_parser import (
    CmdAction,
    InferenceEngineSpec,
    IteratorAction,
    PromptsAction,
)


def _make_yaml(data: dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        yaml.dump(data, f)


def _make_parser_instance() -> MagicMock:
    mp = MagicMock()
    mp.meta = MagicMock()
    mp.tasks = []
    mp.inference_engines = []
    return mp


# ---------------------------------------------------------------------------
# parse_manifest tests
# ---------------------------------------------------------------------------


def _patch_parse(
    monkeypatch: Any,
) -> tuple[MagicMock, MagicMock]:
    parser_cls = MagicMock()
    merge_cls = MagicMock()
    monkeypatch.setattr("tiz.helpers.ManifestParser", parser_cls)
    monkeypatch.setattr("tiz.helpers.merge", merge_cls)
    monkeypatch.setattr("tiz.helpers.can_parallelize", MagicMock(return_value=False))
    return parser_cls, merge_cls


def test_parse_empty_paths(tmp_path: Path, monkeypatch: Any) -> None:
    parser_cls, merge_cls = _patch_parse(monkeypatch)
    mock_manifest = MagicMock()
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_manifest
    merge_cls.assert_called_once_with([])


def test_parse_with_config_yaml(tmp_path: Path, monkeypatch: Any) -> None:
    config_yaml = tmp_path / "config.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, config_yaml)

    parser_instances: list[dict[str, Any]] = []
    parser_cls, merge_cls = _patch_parse(monkeypatch)

    mp = _make_parser_instance()
    mock_manifest = MagicMock()
    mp.get_manifest.return_value = mock_manifest

    def record_parser(
        data: dict[str, Any] | None = None, path: Path | None = None
    ) -> MagicMock:
        parser_instances.append({"data": data, "path": path})
        return mp

    parser_cls.side_effect = record_parser
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_manifest
    assert len(parser_instances) == 1
    assert parser_instances[0]["data"] == {}
    assert parser_instances[0]["path"] == config_yaml
    merge_cls.assert_called_once_with([mock_manifest])


def test_parse_with_config_d(tmp_path: Path, monkeypatch: Any) -> None:
    config_d = tmp_path / "config.d"
    config_d.mkdir()
    c1 = config_d / "a.yaml"
    c2 = config_d / "b.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, c1)
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, c2)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_a = _make_parser_instance()
    parser_b = _make_parser_instance()
    parser_cls.side_effect = [parser_a, parser_b]

    manifest_a = MagicMock()
    manifest_b = MagicMock()
    parser_a.get_manifest.return_value = manifest_a
    parser_b.get_manifest.return_value = manifest_b

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_merged
    parser_cls.assert_has_calls([call(data={}, path=c1), call(data={}, path=c2)])
    merge_cls.assert_called_once_with([manifest_a, manifest_b])


def test_parse_with_config_yaml_and_config_d(tmp_path: Path, monkeypatch: Any) -> None:
    config_yaml = tmp_path / "config.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, config_yaml)

    config_d = tmp_path / "config.d"
    config_d.mkdir()
    c1 = config_d / "a.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, c1)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_a = _make_parser_instance()
    parser_b = _make_parser_instance()
    parser_cls.side_effect = [parser_a, parser_b]

    manifest_a = MagicMock()
    manifest_b = MagicMock()
    parser_a.get_manifest.return_value = manifest_a
    parser_b.get_manifest.return_value = manifest_b

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_merged
    parser_cls.assert_has_calls(
        [call(data={}, path=config_yaml), call(data={}, path=c1)]
    )
    merge_cls.assert_called_once_with([manifest_a, manifest_b])


def test_parse_with_extra_manifests(tmp_path: Path, monkeypatch: Any) -> None:
    extra = tmp_path / "extra.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, extra)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    mp = _make_parser_instance()
    mock_manifest = MagicMock()
    mp.get_manifest.return_value = mock_manifest
    parser_cls.return_value = mp
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [extra])

    assert err is None
    assert manifest is mock_manifest
    parser_cls.assert_called_once_with(data={}, path=extra)
    merge_cls.assert_called_once_with([mock_manifest])


def test_parse_with_config_yaml_and_extra(tmp_path: Path, monkeypatch: Any) -> None:
    config_yaml = tmp_path / "config.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, config_yaml)

    extra = tmp_path / "extra.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, extra)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_a = _make_parser_instance()
    parser_b = _make_parser_instance()
    parser_cls.side_effect = [parser_a, parser_b]

    manifest_a = MagicMock()
    manifest_b = MagicMock()
    parser_a.get_manifest.return_value = manifest_a
    parser_b.get_manifest.return_value = manifest_b

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [extra])

    assert err is None
    assert manifest is mock_merged
    parser_cls.assert_has_calls(
        [call(data={}, path=config_yaml), call(data={}, path=extra)]
    )
    merge_cls.assert_called_once_with([manifest_a, manifest_b])


def test_parse_with_all_sources(tmp_path: Path, monkeypatch: Any) -> None:
    config_yaml = tmp_path / "config.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, config_yaml)

    config_d = tmp_path / "config.d"
    config_d.mkdir()
    c1 = config_d / "a.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, c1)
    c2 = config_d / "b.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 3}}, c2)

    extra = tmp_path / "extra.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 4}}, extra)

    options = {"meta": {"version": "0", "parallelism": 5}}

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parsers = [_make_parser_instance() for _ in range(5)]
    parser_cls.side_effect = parsers

    manifests = [MagicMock() for _ in range(5)]
    for p, m in zip(parsers, manifests, strict=True):
        p.get_manifest.return_value = m

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [extra], options=options)

    assert err is None
    assert manifest is mock_merged
    parser_cls.assert_has_calls(
        [
            call(data={}, path=config_yaml),
            call(data={}, path=c1),
            call(data={}, path=c2),
            call(data={}, path=extra),
            call(data=options, path=None),
        ]
    )
    merge_cls.assert_called_once_with(manifests)


def test_parse_options_after_manifests(tmp_path: Path, monkeypatch: Any) -> None:
    extra = tmp_path / "extra.yaml"
    _make_yaml(
        {
            "meta": {"version": "0", "parallelism": 1},
            "tasks": [{"name": "from_extra"}],
        },
        extra,
    )

    options = {
        "meta": {"version": "0", "parallelism": 2},
        "tasks": [{"name": "from_options"}],
    }

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_extra = _make_parser_instance()
    parser_extra.tasks = [MagicMock()]
    parser_extra.tasks[0].name = "from_extra"

    parser_options = _make_parser_instance()
    parser_options.tasks = [MagicMock()]
    parser_options.tasks[0].name = "from_options"

    parser_cls.side_effect = [parser_extra, parser_options]

    manifest_extra = MagicMock()
    manifest_options = MagicMock()
    parser_extra.get_manifest.return_value = manifest_extra
    parser_options.get_manifest.return_value = manifest_options

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [extra], options=options)

    assert err is None
    assert manifest is mock_merged
    assert merge_cls.call_count == 1
    merge_args = merge_cls.call_args[0]
    assert len(merge_args) == 1
    merged_list = merge_args[0]
    assert len(merged_list) == 2
    assert merged_list[0] == manifest_extra
    assert merged_list[1] == manifest_options


def test_parse_options_none_skips_options_parsing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    parser_cls, merge_cls = _patch_parse(monkeypatch)
    mock_manifest = MagicMock()
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [], options=None)

    assert err is None
    assert manifest is mock_manifest
    merge_cls.assert_called_once_with([])


def test_parse_options_empty_dict_skipped(tmp_path: Path, monkeypatch: Any) -> None:
    parser_cls, merge_cls = _patch_parse(monkeypatch)
    mock_manifest = MagicMock()
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [], options={})

    assert err is None
    assert manifest is mock_manifest
    merge_cls.assert_called_once_with([])


def test_parse_default_options_none_skipped(tmp_path: Path, monkeypatch: Any) -> None:
    parser_cls, merge_cls = _patch_parse(monkeypatch)
    mock_manifest = MagicMock()
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [], default_options=None)

    assert err is None
    assert manifest is mock_manifest
    merge_cls.assert_called_once_with([])


def test_parse_default_options_prepended(tmp_path: Path, monkeypatch: Any) -> None:
    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_default = _make_parser_instance()
    parser_extra = _make_parser_instance()
    parser_cls.side_effect = [parser_default, parser_extra]

    manifest_default = MagicMock()
    manifest_extra = MagicMock()
    parser_default.get_manifest.return_value = manifest_default
    parser_extra.get_manifest.return_value = manifest_extra

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    extra = tmp_path / "extra.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, extra)

    default_opts = {"meta": {"version": "0", "parallelism": 42}}
    manifest, err = parse_manifest(tmp_path, [extra], default_options=default_opts)

    assert err is None
    assert manifest is mock_merged
    parser_cls.assert_has_calls(
        [call(data=default_opts, path=None), call(data={}, path=extra)]
    )
    merge_cls.assert_called_once_with([manifest_default, manifest_extra])


def test_parse_default_options_with_options(tmp_path: Path, monkeypatch: Any) -> None:
    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_default = _make_parser_instance()
    parser_extra = _make_parser_instance()
    parser_opts = _make_parser_instance()
    parser_cls.side_effect = [parser_default, parser_extra, parser_opts]

    manifest_default = MagicMock()
    manifest_extra = MagicMock()
    manifest_opts = MagicMock()
    parser_default.get_manifest.return_value = manifest_default
    parser_extra.get_manifest.return_value = manifest_extra
    parser_opts.get_manifest.return_value = manifest_opts

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    extra = tmp_path / "extra.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, extra)

    default_opts = {"meta": {"version": "0", "parallelism": 42}}
    options = {"meta": {"version": "0", "verbosity": 2}}
    manifest, err = parse_manifest(
        tmp_path, [extra], default_options=default_opts, options=options
    )

    assert err is None
    assert manifest is mock_merged
    parser_cls.assert_has_calls(
        [
            call(data=default_opts, path=None),
            call(data={}, path=extra),
            call(data=options, path=None),
        ]
    )
    merge_cls.assert_called_once_with([manifest_default, manifest_extra, manifest_opts])


def test_parse_config_yaml_not_exists_skipped(tmp_path: Path, monkeypatch: Any) -> None:
    assert not (tmp_path / "config.yaml").exists()

    parser_cls, merge_cls = _patch_parse(monkeypatch)
    mock_manifest = MagicMock()
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_manifest
    merge_cls.assert_called_once_with([])


def test_parse_config_d_not_exists_skipped(tmp_path: Path, monkeypatch: Any) -> None:
    assert not (tmp_path / "config.d").exists()

    parser_cls, merge_cls = _patch_parse(monkeypatch)
    mock_manifest = MagicMock()
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_manifest
    merge_cls.assert_called_once_with([])


def test_parse_multiple_extra_manifests(tmp_path: Path, monkeypatch: Any) -> None:
    extra1 = tmp_path / "extra1.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, extra1)
    extra2 = tmp_path / "extra2.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, extra2)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_a = _make_parser_instance()
    parser_b = _make_parser_instance()
    parser_cls.side_effect = [parser_a, parser_b]

    manifest_a = MagicMock()
    manifest_b = MagicMock()
    parser_a.get_manifest.return_value = manifest_a
    parser_b.get_manifest.return_value = manifest_b

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [extra1, extra2])

    assert err is None
    assert manifest is mock_merged
    assert parser_cls.call_args_list[0] == ({"data": {}, "path": extra1},)
    assert parser_cls.call_args_list[1] == ({"data": {}, "path": extra2},)
    merge_cls.assert_called_once_with([manifest_a, manifest_b])


def test_parse_merge_manifests_in_order(tmp_path: Path, monkeypatch: Any) -> None:
    extra1 = tmp_path / "extra1.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, extra1)
    extra2 = tmp_path / "extra2.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, extra2)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    mp1 = _make_parser_instance()
    mp1.tasks = [MagicMock()]
    mp2 = _make_parser_instance()
    mp2.tasks = [MagicMock()]

    parser_cls.side_effect = [mp1, mp2]

    m1 = MagicMock()
    m2 = MagicMock()
    mp1.get_manifest.return_value = m1
    mp2.get_manifest.return_value = m2

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [extra1, extra2])

    assert err is None
    assert manifest is mock_merged
    assert merge_cls.call_count == 1
    merge_args = merge_cls.call_args[0]
    assert len(merge_args) == 1
    merged_list = merge_args[0]
    assert len(merged_list) == 2
    assert merged_list[0] == m1
    assert merged_list[1] == m2


def test_parse_extra_manifests_after_config_d(tmp_path: Path, monkeypatch: Any) -> None:
    config_d = tmp_path / "config.d"
    config_d.mkdir()
    c1 = config_d / "a.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, c1)

    extra = tmp_path / "extra.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, extra)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_a = _make_parser_instance()
    parser_b = _make_parser_instance()
    parser_cls.side_effect = [parser_a, parser_b]

    manifest_a = MagicMock()
    manifest_b = MagicMock()
    parser_a.get_manifest.return_value = manifest_a
    parser_b.get_manifest.return_value = manifest_b

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [extra])

    assert err is None
    assert manifest is mock_merged
    assert parser_cls.call_args_list[0] == ({"data": {}, "path": c1},)
    assert parser_cls.call_args_list[1] == ({"data": {}, "path": extra},)
    merge_cls.assert_called_once_with([manifest_a, manifest_b])


def test_parse_nested_subdirs_in_config_d_ignored(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config_d = tmp_path / "config.d"
    config_d.mkdir()
    nested = config_d / "subdir"
    nested.mkdir()
    deep = nested / "deep.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, deep)

    c1 = config_d / "a.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, c1)

    parser_instances: list[dict[str, Any]] = []
    parser_cls, merge_cls = _patch_parse(monkeypatch)

    mp = _make_parser_instance()
    mock_manifest = MagicMock()
    mp.get_manifest.return_value = mock_manifest

    def record_parser(
        data: dict[str, Any] | None = None, path: Path | None = None
    ) -> MagicMock:
        parser_instances.append({"data": data, "path": path})
        return mp

    parser_cls.side_effect = record_parser
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_manifest
    assert len(parser_instances) == 1
    assert parser_instances[0]["path"] == c1


def test_parse_config_d_sorted_order(tmp_path: Path, monkeypatch: Any) -> None:
    config_d = tmp_path / "config.d"
    config_d.mkdir()
    c_b = config_d / "b.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, c_b)
    c_a = config_d / "a.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, c_a)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_a = _make_parser_instance()
    parser_b = _make_parser_instance()
    parser_cls.side_effect = [parser_a, parser_b]

    manifest_a = MagicMock()
    manifest_b = MagicMock()
    parser_a.get_manifest.return_value = manifest_a
    parser_b.get_manifest.return_value = manifest_b

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_merged
    assert parser_cls.call_args_list[0] == ({"data": {}, "path": c_a},)
    assert parser_cls.call_args_list[1] == ({"data": {}, "path": c_b},)
    merge_cls.assert_called_once_with([manifest_a, manifest_b])


def test_parse_config_precedes_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    config_yaml = tmp_path / "config.yaml"
    _make_yaml(
        {
            "meta": {"version": "0", "parallelism": 1},
            "tasks": [{"name": "from_config"}],
        },
        config_yaml,
    )

    extra = tmp_path / "extra.yaml"
    _make_yaml(
        {
            "meta": {"version": "0", "parallelism": 2},
            "tasks": [{"name": "from_extra"}],
        },
        extra,
    )

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_a = _make_parser_instance()
    parser_a.tasks = [MagicMock()]
    parser_a.tasks[0].name = "from_config"

    parser_b = _make_parser_instance()
    parser_b.tasks = [MagicMock()]
    parser_b.tasks[0].name = "from_extra"

    parser_cls.side_effect = [parser_a, parser_b]

    manifest_a = MagicMock()
    manifest_b = MagicMock()
    parser_a.get_manifest.return_value = manifest_a
    parser_b.get_manifest.return_value = manifest_b

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [extra])

    assert err is None
    assert manifest is mock_merged
    assert merge_cls.call_count == 1
    merge_args = merge_cls.call_args[0]
    assert len(merge_args) == 1
    merged_list = merge_args[0]
    assert len(merged_list) == 2
    assert merged_list[0] == manifest_a
    assert merged_list[1] == manifest_b


def test_parse_config_d_only_yaml_files(tmp_path: Path, monkeypatch: Any) -> None:
    config_d = tmp_path / "config.d"
    config_d.mkdir()
    c1 = config_d / "a.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, c1)

    other = config_d / "not_a_yaml.txt"
    other.write_text("not yaml")

    parser_instances: list[dict[str, Any]] = []
    parser_cls, merge_cls = _patch_parse(monkeypatch)

    mp = _make_parser_instance()
    mock_manifest = MagicMock()
    mp.get_manifest.return_value = mock_manifest

    def record_parser(
        data: dict[str, Any] | None = None, path: Path | None = None
    ) -> MagicMock:
        parser_instances.append({"data": data, "path": path})
        return mp

    parser_cls.side_effect = record_parser
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_manifest
    assert len(parser_instances) == 1
    assert parser_instances[0]["path"] == c1


def test_parse_empty_config_d(tmp_path: Path, monkeypatch: Any) -> None:
    config_d = tmp_path / "config.d"
    config_d.mkdir()

    parser_cls, merge_cls = _patch_parse(monkeypatch)
    mock_manifest = MagicMock()
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is mock_manifest
    merge_cls.assert_called_once_with([])


def test_parse_relative_manifest_falls_back_to_manifests_dir(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    rel = Path("extra.yaml")
    abs_path = manifests_dir / rel
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, abs_path)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    mp = _make_parser_instance()
    mock_manifest = MagicMock()
    mp.get_manifest.return_value = mock_manifest
    parser_cls.return_value = mp
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [rel])

    assert err is None
    assert manifest is mock_manifest
    parser_cls.assert_called_once_with(data={}, path=abs_path)
    merge_cls.assert_called_once_with([mock_manifest])


def test_parse_relative_manifest_non_existent_falls_back_to_manifests_dir(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    rel = Path("nonexistent.yaml")
    abs_path = manifests_dir / rel

    parser_cls, merge_cls = _patch_parse(monkeypatch)
    mock_manifest = MagicMock()
    mp = _make_parser_instance()
    mp.get_manifest.return_value = mock_manifest
    parser_cls.return_value = mp
    merge_cls.return_value = mock_manifest

    manifest, err = parse_manifest(tmp_path, [rel])

    assert err is None
    assert manifest is mock_manifest
    parser_cls.assert_called_once_with(data={}, path=abs_path)
    merge_cls.assert_called_once_with([mock_manifest])


def test_parse_parser_error_returns_error(tmp_path: Path, monkeypatch: Any) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text("broken: [unclosed", encoding="utf-8")

    monkeypatch.setattr("tiz.helpers.merge", MagicMock())

    manifest, err = parse_manifest(tmp_path, [extra])

    assert manifest is None
    assert err is not None
    assert "Manifest parsing/merge error" in err


def test_parse_merge_error_on_empty(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("tiz.helpers.ManifestParser", MagicMock())

    manifest, err = parse_manifest(tmp_path, [])

    assert manifest is None
    assert err is not None
    assert "Manifest parsing/merge error" in err


def test_parse_parser_error_empty_options_no_manifests(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "tiz.helpers.ManifestParser", MagicMock(side_effect=ValueError("parser fail"))
    )
    monkeypatch.setattr("tiz.helpers.merge", MagicMock())

    manifest, err = parse_manifest(
        tmp_path, [], options={"meta": {"version": "0", "parallelism": 1}}
    )

    assert manifest is None
    assert err is not None


def test_parse_with_config_yaml_and_config_d_and_extra_manifests(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config_yaml = tmp_path / "config.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 1}}, config_yaml)

    config_d = tmp_path / "config.d"
    config_d.mkdir()
    c1 = config_d / "a.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 2}}, c1)

    extra = tmp_path / "extra.yaml"
    _make_yaml({"meta": {"version": "0", "parallelism": 3}}, extra)

    parser_cls, merge_cls = _patch_parse(monkeypatch)

    parser_a = _make_parser_instance()
    parser_b = _make_parser_instance()
    parser_c = _make_parser_instance()
    parser_cls.side_effect = [parser_a, parser_b, parser_c]

    manifest_a = MagicMock()
    manifest_b = MagicMock()
    manifest_c = MagicMock()
    parser_a.get_manifest.return_value = manifest_a
    parser_b.get_manifest.return_value = manifest_b
    parser_c.get_manifest.return_value = manifest_c

    mock_merged = MagicMock()
    merge_cls.return_value = mock_merged

    manifest, err = parse_manifest(tmp_path, [extra])

    assert err is None
    assert manifest is mock_merged
    assert parser_cls.call_args_list[0] == ({"data": {}, "path": config_yaml},)
    assert parser_cls.call_args_list[1] == ({"data": {}, "path": c1},)
    assert parser_cls.call_args_list[2] == ({"data": {}, "path": extra},)
    merge_cls.assert_called_once_with([manifest_a, manifest_b, manifest_c])


def test_parse_keeps_parallelism_when_parallelizable(
    tmp_path: Path,
) -> None:
    """When can_parallelize returns True, parallelism is left unchanged (>1)."""
    config_yaml = tmp_path / "config.yaml"
    # Two consecutive tasks with allow_parallel_run=True and parallelism > 1
    _make_yaml(
        {
            "meta": {"version": "0", "parallelism": 2},
            "tasks": [
                {"name": "task1", "allow_parallel_run": True},
                {"name": "task2", "allow_parallel_run": True},
            ],
        },
        config_yaml,
    )

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is not None
    assert manifest.meta.parallelism == 2


def test_parse_forces_parallelism_to_1_when_not_parallelizable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When can_parallelize returns False, parallelism is forced to 1."""
    config_yaml = tmp_path / "config.yaml"
    # Single task with parallelism > 1 but no parallelizable content
    _make_yaml(
        {
            "meta": {"version": "0", "parallelism": 2},
            "tasks": [
                {
                    "name": "task1",
                    "allow_parallel_run": False,
                    "actions": [{"cmd": "sync"}],
                }
            ],
        },
        config_yaml,
    )

    manifest, err = parse_manifest(tmp_path, [])

    assert err is None
    assert manifest is not None
    assert manifest.meta.parallelism == 1
    assert "no consecutive parallel tasks" in caplog.text


# ---------------------------------------------------------------------------
# run tests
# ---------------------------------------------------------------------------


def _patch_run(monkeypatch: Any) -> tuple[MagicMock, Any]:
    executor_cls = MagicMock()
    executor_instance = MagicMock()
    executor_instance.task_usage = {}
    executor_cls.return_value = executor_instance
    monkeypatch.setattr("tiz.helpers.ManifestExecutor", executor_cls)
    return executor_cls, executor_instance


def test_run_success(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    executor_cls, executor_instance = _patch_run(monkeypatch)

    usage, err = run(manifest=mock_manifest, base_path=tmp_path)

    assert err is None
    assert usage == {}
    executor_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        update_callback=None,
        context=None,
        confirm_callback=None,
    )
    executor_instance.execute.assert_called_once_with()


def test_run_with_update_callback(tmp_path: Path, monkeypatch: Any) -> None:
    cb = MagicMock()
    mock_manifest = MagicMock()
    executor_cls, executor_instance = _patch_run(monkeypatch)

    usage, err = run(manifest=mock_manifest, base_path=tmp_path, update_callback=cb)

    assert err is None
    assert usage == {}
    executor_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        update_callback=cb,
        context=None,
        confirm_callback=None,
    )
    executor_instance.execute.assert_called_once_with()


def test_run_with_context(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    executor_cls, executor_instance = _patch_run(monkeypatch)
    ctx = {"key": "value"}

    usage, err = run(manifest=mock_manifest, base_path=tmp_path, context=ctx)

    assert err is None
    assert usage == {}
    executor_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        update_callback=None,
        context=ctx,
        confirm_callback=None,
    )
    executor_instance.execute.assert_called_once_with()


def test_run_executor_error_returns_usage_and_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    mock_manifest = MagicMock()
    mock_manifest.meta.verbosity = 0
    executor_cls, executor_instance = _patch_run(monkeypatch)
    executor_instance.execute.side_effect = RuntimeError("execution failed")
    executor_instance.task_usage = {"task1": {"prompt_tokens": 10, "prompt_time": 0.1}}

    usage, err = run(manifest=mock_manifest, base_path=tmp_path)

    assert err is not None
    assert "Manifest execution error" in err
    assert usage == {"task1": {"prompt_tokens": 10, "prompt_time": 0.1}}


def test_run_keyboard_interrupt(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    executor_cls, executor_instance = _patch_run(monkeypatch)
    executor_instance.execute.side_effect = KeyboardInterrupt
    executor_instance.task_usage = {"task1": {"prompt_tokens": 5, "prompt_time": 0.2}}

    usage, err = run(manifest=mock_manifest, base_path=tmp_path)

    assert err == "interrupted"
    assert usage == {"task1": {"prompt_tokens": 5, "prompt_time": 0.2}}


def test_run_constructor_keyboard_interrupt(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    executor_cls = MagicMock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr("tiz.helpers.ManifestExecutor", executor_cls)

    usage, err = run(manifest=mock_manifest, base_path=tmp_path)

    assert err == "interrupted"
    assert usage == {}


def test_run_constructor_exception(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    mock_manifest.meta.verbosity = 0
    executor_cls = MagicMock(side_effect=ValueError("constructor fail"))
    monkeypatch.setattr("tiz.helpers.ManifestExecutor", executor_cls)

    usage, err = run(manifest=mock_manifest, base_path=tmp_path)

    assert err is not None
    assert "Manifest execution error" in err
    assert usage == {}


# ---------------------------------------------------------------------------
# chat tests
# ---------------------------------------------------------------------------


def _patch_chat(monkeypatch: Any) -> tuple[MagicMock, MagicMock]:
    chat_cls = MagicMock()
    chat_instance = MagicMock()
    chat_instance.task_usage = {}
    chat_cls.return_value = chat_instance
    monkeypatch.setattr("tiz.helpers.InteractiveChat", chat_cls)
    return chat_cls, chat_instance


def test_chat_success(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    chat_cls, chat_instance = _patch_chat(monkeypatch)

    usage, err = chat(manifest=mock_manifest, base_path=tmp_path)

    assert err is None
    assert usage == {}
    chat_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        task_name=None,
        update_callback=None,
        input_callback=None,
        context=None,
        confirm_callback=None,
        enable_recording=False,
    )
    chat_instance.run.assert_called_once_with()


def test_chat_with_callbacks(tmp_path: Path, monkeypatch: Any) -> None:
    cb = MagicMock()
    input_cb = MagicMock(return_value="test input")
    mock_manifest = MagicMock()
    chat_cls, chat_instance = _patch_chat(monkeypatch)

    usage, err = chat(
        manifest=mock_manifest,
        base_path=tmp_path,
        update_callback=cb,
        input_callback=input_cb,
    )

    assert err is None
    assert usage == {}
    chat_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        task_name=None,
        update_callback=cb,
        input_callback=input_cb,
        context=None,
        confirm_callback=None,
        enable_recording=False,
    )
    chat_instance.run.assert_called_once_with()


def test_chat_with_task_name(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    chat_cls, chat_instance = _patch_chat(monkeypatch)

    usage, err = chat(manifest=mock_manifest, base_path=tmp_path, task_name="my_task")

    assert err is None
    assert usage == {}
    chat_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        task_name="my_task",
        update_callback=None,
        input_callback=None,
        context=None,
        confirm_callback=None,
        enable_recording=False,
    )
    chat_instance.run.assert_called_once_with()


def test_chat_with_context(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    chat_cls, chat_instance = _patch_chat(monkeypatch)
    ctx = {"key": "value"}

    usage, err = chat(
        manifest=mock_manifest,
        base_path=tmp_path,
        context=ctx,
    )

    assert err is None
    assert usage == {}
    chat_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        task_name=None,
        update_callback=None,
        input_callback=None,
        context=ctx,
        confirm_callback=None,
        enable_recording=False,
    )
    chat_instance.run.assert_called_once_with()


def test_chat_error_returns_usage_and_error(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    mock_manifest.meta.verbosity = 0
    chat_cls, chat_instance = _patch_chat(monkeypatch)
    chat_instance.run.side_effect = RuntimeError("chat failed")
    chat_instance.task_usage = {"task1": {"prompt_tokens": 10, "prompt_time": 0.1}}

    usage, err = chat(manifest=mock_manifest, base_path=tmp_path)

    assert err is not None
    assert "Interactive chat error" in err
    assert usage == {"task1": {"prompt_tokens": 10, "prompt_time": 0.1}}


def test_chat_keyboard_interrupt(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    chat_cls, chat_instance = _patch_chat(monkeypatch)
    chat_instance.run.side_effect = KeyboardInterrupt
    chat_instance.task_usage = {"task1": {"prompt_tokens": 5, "prompt_time": 0.2}}

    usage, err = chat(manifest=mock_manifest, base_path=tmp_path)

    assert err == "interrupted"
    assert usage == {"task1": {"prompt_tokens": 5, "prompt_time": 0.2}}


def test_chat_constructor_keyboard_interrupt(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    chat_cls = MagicMock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr("tiz.helpers.InteractiveChat", chat_cls)

    usage, err = chat(manifest=mock_manifest, base_path=tmp_path)

    assert err == "interrupted"
    assert usage == {}


def test_chat_constructor_exception(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    mock_manifest.meta.verbosity = 0
    chat_cls = MagicMock(side_effect=ValueError("constructor fail"))
    monkeypatch.setattr("tiz.helpers.InteractiveChat", chat_cls)

    usage, err = chat(manifest=mock_manifest, base_path=tmp_path)

    assert err is not None
    assert "Interactive chat error" in err
    assert usage == {}


def test_chat_with_confirm_callback(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    chat_cls, chat_instance = _patch_chat(monkeypatch)
    confirm_cb = MagicMock(return_value=True)

    usage, err = chat(
        manifest=mock_manifest,
        base_path=tmp_path,
        confirm_callback=confirm_cb,
    )

    assert err is None
    assert usage == {}
    chat_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        task_name=None,
        update_callback=None,
        input_callback=None,
        context=None,
        confirm_callback=confirm_cb,
        enable_recording=False,
    )
    chat_instance.run.assert_called_once_with()


def test_chat_with_enable_recording(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    chat_cls, chat_instance = _patch_chat(monkeypatch)

    usage, err = chat(
        manifest=mock_manifest,
        base_path=tmp_path,
        enable_recording=True,
    )

    assert err is None
    assert usage == {}
    chat_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        task_name=None,
        update_callback=None,
        input_callback=None,
        context=None,
        confirm_callback=None,
        enable_recording=True,
    )
    chat_instance.run.assert_called_once_with()


def test_run_with_confirm_callback(tmp_path: Path, monkeypatch: Any) -> None:
    mock_manifest = MagicMock()
    executor_cls, executor_instance = _patch_run(monkeypatch)
    confirm_cb = MagicMock(return_value=True)

    usage, err = run(
        manifest=mock_manifest,
        base_path=tmp_path,
        confirm_callback=confirm_cb,
    )

    assert err is None
    assert usage == {}
    executor_cls.assert_called_once_with(
        manifest=mock_manifest,
        base_path=tmp_path,
        update_callback=None,
        context=None,
        confirm_callback=confirm_cb,
    )
    executor_instance.execute.assert_called_once_with()


# ---------------------------------------------------------------------------
# Tests for exec_cmd
# ---------------------------------------------------------------------------


def _make_task(**kwargs: Any) -> MagicMock:
    task = MagicMock()
    task.name = kwargs.get("name", "test_task")
    task.worker_image = kwargs.get("worker_image", "ubuntu")
    task.project = kwargs.get("project")
    task.force_copy_files = kwargs.get("force_copy_files", [])
    task.readonly_sandbox = kwargs.get("readonly_sandbox", False)
    task.allow_parallel_run = kwargs.get("allow_parallel_run", False)
    task.actions = kwargs.get("actions", [])
    return task


def _make_manifest(tasks: list[MagicMock] | None = None) -> MagicMock:
    manifest = MagicMock()
    manifest.tasks = tasks or []
    manifest.meta.container_engine = None
    manifest.meta.verbosity = 0
    manifest.meta.use_host_timezone = True
    manifest.meta.committer_name = None
    manifest.meta.committer_email = None
    return manifest


def test_exec_cmd_no_tasks() -> None:
    manifest = _make_manifest(tasks=[])
    result = exec_cmd(manifest=manifest, base_path=Path("/tmp"))
    assert result == "Error: no tasks found in manifests"


def test_exec_cmd_task_not_found() -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])
    result = exec_cmd(
        manifest=manifest,
        base_path=Path("/tmp"),
        task_name="nonexistent",
    )
    assert result == "Error: task 'nonexistent' not found in manifests"


def test_exec_cmd_no_engine(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])
    manifest.meta.container_engine = None

    monkeypatch.setattr("tiz.helpers.SandboxManager.available_engine", lambda: None)

    result = exec_cmd(manifest=manifest, base_path=tmp_path)
    assert result == "Error: no container engine (podman or docker) found"


def test_exec_cmd_sandbox_create_error(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)
    mock_manager.create_sandbox.side_effect = RuntimeError("sandbox fail")

    result = exec_cmd(manifest=manifest, base_path=tmp_path)

    assert result is not None
    assert "Error creating sandbox: sandbox fail" in result
    assert "helpers.py" in result
    manager_cls.assert_called_once_with(base_path=tmp_path, engine="docker")
    mock_manager.create_sandbox.assert_called_once_with(
        sandbox_name="my_task",
        project_path=None,
        force_copy_files=None,
        committer_name="Tiz",
        committer_email="tiz@example.com",
    )


def test_exec_cmd_container_create_error(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)
    mock_manager.create_container.side_effect = RuntimeError("container fail")

    result = exec_cmd(manifest=manifest, base_path=tmp_path)

    assert result is not None
    assert "Error creating container: container fail" in result
    assert "helpers.py" in result
    mock_manager.create_container.assert_called_once()
    # sandbox must be cleaned up on container creation failure
    mock_manager.kill_and_delete_sandbox.assert_called_once_with("my_task")


def test_exec_cmd_container_id_none(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = None
    mock_manager.create_container.return_value = mock_container

    result = exec_cmd(manifest=manifest, base_path=tmp_path)

    assert result == "Error: container ID is None"
    assert mock_manager.create_container.called
    mock_manager.kill_and_delete_sandbox.assert_called_once_with("my_task")


def test_exec_cmd_success(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "abc123"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(
        manifest=manifest,
        base_path=tmp_path,
        task_name="my_task",
        cmd_args=["echo", "hello"],
    )

    assert result is None
    manager_cls.assert_called_once_with(base_path=tmp_path, engine="docker")
    mock_manager.create_sandbox.assert_called_once_with(
        sandbox_name="my_task",
        project_path=None,
        force_copy_files=None,
        committer_name="Tiz",
        committer_email="tiz@example.com",
    )
    mock_manager.create_container.assert_called_once_with(
        sandbox_name="my_task",
        image="ubuntu",
        network="internet",
        read_only_project=False,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
    )
    popen.assert_called_once_with(["docker", "exec", "-it", "abc123", "echo", "hello"])
    mock_proc.wait.assert_called_once_with()
    mock_container.stop.assert_called_once_with(timeout=0)


def test_exec_cmd_success_default_task(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="first_task", worker_image="alpine")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "def456"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(manifest=manifest, base_path=tmp_path)

    assert result is None

    mock_manager.create_sandbox.assert_called_once_with(
        sandbox_name="first_task",
        project_path=None,
        force_copy_files=None,
        committer_name="Tiz",
        committer_email="tiz@example.com",
    )
    popen.assert_called_once_with(
        ["docker", "exec", "-it", "def456", "/bin/bash", "-l"]
    )
    mock_container.stop.assert_called_once_with(timeout=0)


def test_exec_cmd_success_with_project_and_force_copy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    task = _make_task(
        name="my_task",
        worker_image="ubuntu",
        project="/some/project",
        force_copy_files=["file1.txt", "file2.txt"],
        readonly_sandbox=True,
    )
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "ghi789"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(manifest=manifest, base_path=tmp_path)

    assert result is None

    mock_manager.create_sandbox.assert_called_once_with(
        sandbox_name="my_task",
        project_path="/some/project",
        force_copy_files=["file1.txt", "file2.txt"],
        committer_name="Tiz",
        committer_email="tiz@example.com",
    )
    mock_manager.create_container.assert_called_once_with(
        sandbox_name="my_task",
        image="ubuntu",
        network="internet",
        read_only_project=True,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
    )


def test_exec_cmd_success_with_project_no_force_copy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """project set but force_copy_files is empty list (becomes None via or None)."""
    task = _make_task(
        name="my_task",
        worker_image="ubuntu",
        project="/some/project",
        force_copy_files=[],
    )
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "ghi789"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(manifest=manifest, base_path=tmp_path)

    assert result is None

    mock_manager.create_sandbox.assert_called_once_with(
        sandbox_name="my_task",
        project_path="/some/project",
        force_copy_files=None,
        committer_name="Tiz",
        committer_email="tiz@example.com",
    )
    mock_container.stop.assert_called_once_with(timeout=0)


def test_exec_cmd_verbosity_none(tmp_path: Path, monkeypatch: Any) -> None:
    """verbosity is None, defaulting to 0 for verbose param."""
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])
    manifest.meta.verbosity = None

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "abc123"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(manifest=manifest, base_path=tmp_path)

    assert result is None
    mock_manager.create_container.assert_called_once_with(
        sandbox_name="my_task",
        image="ubuntu",
        network="internet",
        read_only_project=False,
        extra_run_args=None,
        verbose=0,
        use_host_timezone=True,
    )


def test_exec_cmd_success_returncode_nonzero(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "abc123"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.returncode = 42
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(manifest=manifest, base_path=tmp_path)

    assert result is None
    mock_proc.wait.assert_called_once_with()
    mock_container.stop.assert_called_once_with(timeout=0)


def test_exec_cmd_task_name_whitespace(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="My Complex Task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "abc123"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(
        manifest=manifest, base_path=tmp_path, task_name="My Complex Task"
    )

    assert result is None

    mock_manager.create_sandbox.assert_called_once_with(
        sandbox_name="my_complex_task",
        project_path=None,
        force_copy_files=None,
        committer_name="Tiz",
        committer_email="tiz@example.com",
    )
    mock_container.stop.assert_called_once_with(timeout=0)


def test_exec_cmd_keyboard_interrupt(tmp_path: Path, monkeypatch: Any) -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "abc123"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.wait.side_effect = [KeyboardInterrupt, None]
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(
        manifest=manifest,
        base_path=tmp_path,
        task_name="my_task",
        cmd_args=["echo", "hello"],
    )

    assert result == "interrupted"
    mock_proc.terminate.assert_called_once_with()
    assert mock_proc.wait.call_count == 2
    mock_container.stop.assert_called_once_with(timeout=0)


def test_exec_cmd_keyboard_interrupt_process_already_exited(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """KeyboardInterrupt fires but the process already exited (poll returns non-None)."""
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "abc123"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    # Process already exited (poll returns 0)
    mock_proc.poll.return_value = 0
    mock_proc.wait.side_effect = KeyboardInterrupt
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    result = exec_cmd(
        manifest=manifest,
        base_path=tmp_path,
        task_name="my_task",
        cmd_args=["echo", "hello"],
    )

    assert result == "interrupted"
    # terminate should NOT be called since process already exited
    mock_proc.terminate.assert_not_called()
    assert mock_proc.wait.call_count == 1
    mock_container.stop.assert_called_once_with(timeout=0)


def test_exec_cmd_extra_run_args(tmp_path: Path, monkeypatch: Any) -> None:
    """Test that extra_run_args are passed to create_container."""
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "podman"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)

    mock_container = MagicMock()
    mock_container.container_id = "abc123"
    mock_manager.create_container.return_value = mock_container

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    popen = MagicMock(return_value=mock_proc)
    monkeypatch.setattr("tiz.helpers.subprocess.Popen", popen)

    extra_args = ["--cpus=2", "--memory=512m"]
    result = exec_cmd(
        manifest=manifest,
        base_path=tmp_path,
        task_name="my_task",
        cmd_args=["bash"],
        extra_run_args=extra_args,
    )

    assert result is None
    mock_manager.create_container.assert_called_once_with(
        sandbox_name="my_task",
        image="ubuntu",
        network="internet",
        read_only_project=False,
        extra_run_args=extra_args,
        verbose=0,
        use_host_timezone=True,
    )
    mock_container.stop.assert_called_once_with(timeout=0)


# ---------------------------------------------------------------------------
# _format_exc
# ---------------------------------------------------------------------------


def test_format_exc_verbosity_0() -> None:
    try:
        raise RuntimeError("test error")
    except RuntimeError as exc:
        result = _format_exc(exc, verbosity=0)
    assert "test error" in result
    assert "helpers.py" in result


def test_format_exc_verbosity_2() -> None:
    try:
        raise RuntimeError("test error")
    except RuntimeError as exc:
        result = _format_exc(exc, verbosity=2)
    assert "test error" in result
    assert "RuntimeError" in result
    assert "Traceback" in result


def test_format_exc_no_tb_frames() -> None:
    exc = RuntimeError("bare error")
    result = _format_exc(exc, verbosity=0)
    assert result == "bare error"


def test_format_exc_with_tiz_in_filename() -> None:
    """Frame with 'tiz' in filename is kept (condition short-circuits)."""
    import traceback

    try:
        raise RuntimeError("test")
    except RuntimeError as exc:
        with patch("tiz.helpers.traceback.extract_tb") as mock_extract:
            frame = traceback.FrameSummary(
                "/home/user/tiz/tools/bash.py", 50, "bash_cmd"
            )
            mock_extract.return_value = [frame]
            result = _format_exc(exc, verbosity=0)
    assert "tiz/tools/bash.py:50" in result
    assert "test" in result


def test_format_exc_all_frames_from_site_packages() -> None:
    """All frames from site-packages, project_frames falls back to tb[-1:] for last frame."""
    import traceback

    try:
        raise RuntimeError("site-packages error")
    except RuntimeError as exc:
        with patch("tiz.helpers.traceback.extract_tb") as mock_extract:
            frame = traceback.FrameSummary(
                "/usr/lib/python3.12/site-packages/foo/bar.py", 42, "func"
            )
            mock_extract.return_value = [frame]
            result = _format_exc(exc, verbosity=0)
    assert "site-packages error" in result
    assert "bar.py:42" in result


def test_exec_cmd_sandbox_create_error_verbosity_2(
    tmp_path: Path, monkeypatch: Any
) -> None:
    task = _make_task(name="my_task")
    manifest = _make_manifest(tasks=[task])
    manifest.meta.verbosity = 2

    mock_manager = MagicMock()
    manager_cls = MagicMock(return_value=mock_manager)
    manager_cls.available_engine.return_value = "docker"
    monkeypatch.setattr("tiz.helpers.SandboxManager", manager_cls)
    mock_manager.create_sandbox.side_effect = RuntimeError("sandbox fail")

    result = exec_cmd(manifest=manifest, base_path=tmp_path)
    assert result is not None
    assert "Error creating sandbox: " in result
    assert "RuntimeError" in result


# ---------------------------------------------------------------------------
# get_credits tests
# ---------------------------------------------------------------------------


def test_get_credits_empty_engines() -> None:
    result = get_credits([])
    assert result == []


def test_get_credits_single_engine(monkeypatch: Any) -> None:
    mock_engine = MagicMock(spec=InferenceEngineSpec)
    mock_engine.name = "test-engine"

    mock_client = MagicMock()
    mock_client.get_credits.return_value = {
        "total_credits": 100.0,
        "total_usage": 25.0,
    }
    monkeypatch.setattr(
        "tiz.helpers.BaseTaskExecutor.build_client",
        MagicMock(return_value=mock_client),
    )

    result = get_credits([mock_engine])
    assert len(result) == 1
    assert result[0]["name"] == "test-engine"
    assert result[0]["total_credits"] == 100.0
    assert result[0]["total_usage"] == 25.0
    assert result[0]["remaining"] == 75.0


def test_get_credits_multiple_engines(monkeypatch: Any) -> None:
    mock_engine1 = MagicMock(spec=InferenceEngineSpec)
    mock_engine1.name = "engine1"
    mock_engine2 = MagicMock(spec=InferenceEngineSpec)
    mock_engine2.name = "engine2"

    mock_client1 = MagicMock()
    mock_client1.get_credits.return_value = {
        "total_credits": 50.0,
        "total_usage": 10.0,
    }
    mock_client2 = MagicMock()
    mock_client2.get_credits.return_value = {
        "total_credits": 200.0,
        "total_usage": 50.0,
    }
    monkeypatch.setattr(
        "tiz.helpers.BaseTaskExecutor.build_client",
        MagicMock(side_effect=[mock_client1, mock_client2]),
    )

    result = get_credits([mock_engine1, mock_engine2])
    assert len(result) == 2
    assert result[0]["name"] == "engine1"
    assert result[0]["total_credits"] == 50.0
    assert result[0]["total_usage"] == 10.0
    assert result[0]["remaining"] == 40.0
    assert result[1]["name"] == "engine2"
    assert result[1]["total_credits"] == 200.0
    assert result[1]["total_usage"] == 50.0
    assert result[1]["remaining"] == 150.0


def test_get_credits_missing_keys(monkeypatch: Any) -> None:
    mock_engine = MagicMock(spec=InferenceEngineSpec)
    mock_engine.name = "missing-engine"

    mock_client = MagicMock()
    mock_client.get_credits.return_value = {}
    monkeypatch.setattr(
        "tiz.helpers.BaseTaskExecutor.build_client",
        MagicMock(return_value=mock_client),
    )

    result = get_credits([mock_engine])
    assert len(result) == 1
    assert result[0]["name"] == "missing-engine"
    assert result[0]["total_credits"] == 0.0
    assert result[0]["total_usage"] == 0.0
    assert result[0]["remaining"] == 0.0


# ---------------------------------------------------------------------------
# ask_confirmations tests
# ---------------------------------------------------------------------------


def _make_tool_spec(
    name: str = "tool1", confirmations: list | None = None
) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.confirmations = confirmations or []
    return tool


def _make_subagent_spec(
    name: str = "sub1", tools: list[MagicMock] | None = None
) -> MagicMock:
    sub = MagicMock()
    sub.name = name
    sub.tools = tools or []
    return sub


def test_ask_confirmations_no_tasks() -> None:
    manifest = _make_manifest(tasks=[])
    assert ask_confirmations(manifest) is False


def test_ask_confirmations_no_tools() -> None:
    task = _make_task(name="task1")
    task.tools = []
    task.subagents = []
    manifest = _make_manifest(tasks=[task])
    assert ask_confirmations(manifest) is False


def test_ask_confirmations_tool_no_confirmations() -> None:
    tool = _make_tool_spec("web", confirmations=[])
    task = _make_task(name="task1")
    task.tools = [tool]
    task.subagents = []
    manifest = _make_manifest(tasks=[task])
    assert ask_confirmations(manifest) is False


def test_ask_confirmations_tool_with_confirmations() -> None:
    confirmation = MagicMock()
    confirmation.type = "any"
    tool = _make_tool_spec("web", confirmations=[confirmation])
    task = _make_task(name="task1")
    task.tools = [tool]
    task.subagents = []
    manifest = _make_manifest(tasks=[task])
    assert ask_confirmations(manifest) is True


def test_ask_confirmations_multiple_tools_one_has_confirmations() -> None:
    tool1 = _make_tool_spec("web", confirmations=[])
    confirmation = MagicMock()
    confirmation.type = "any"
    tool2 = _make_tool_spec("shell", confirmations=[confirmation])
    task = _make_task(name="task1")
    task.tools = [tool1, tool2]
    task.subagents = []
    manifest = _make_manifest(tasks=[task])
    assert ask_confirmations(manifest) is True


def test_ask_confirmations_subagent_tool_with_confirmations() -> None:
    confirmation = MagicMock()
    confirmation.type = "any"
    tool = _make_tool_spec("web", confirmations=[confirmation])
    subagent = _make_subagent_spec("sub1", tools=[tool])
    task = _make_task(name="task1")
    task.tools = []
    task.subagents = [subagent]
    manifest = _make_manifest(tasks=[task])
    assert ask_confirmations(manifest) is True


def test_ask_confirmations_subagent_tool_no_confirmations() -> None:
    tool = _make_tool_spec("web", confirmations=[])
    subagent = _make_subagent_spec("sub1", tools=[tool])
    task = _make_task(name="task1")
    task.tools = []
    task.subagents = [subagent]
    manifest = _make_manifest(tasks=[task])
    assert ask_confirmations(manifest) is False


def test_ask_confirmations_multiple_tasks_first_has_confirmations() -> None:
    confirmation = MagicMock()
    confirmation.type = "any"
    tool = _make_tool_spec("web", confirmations=[confirmation])
    task1 = _make_task(name="task1")
    task1.tools = [tool]
    task1.subagents = []
    task2 = _make_task(name="task2")
    task2.tools = []
    task2.subagents = []
    manifest = _make_manifest(tasks=[task1, task2])
    assert ask_confirmations(manifest) is True


def test_ask_confirmations_multiple_tasks_second_has_confirmations() -> None:
    task1 = _make_task(name="task1")
    task1.tools = []
    task1.subagents = []
    confirmation = MagicMock()
    confirmation.type = "any"
    tool = _make_tool_spec("web", confirmations=[confirmation])
    task2 = _make_task(name="task2")
    task2.tools = [tool]
    task2.subagents = []
    manifest = _make_manifest(tasks=[task1, task2])
    assert ask_confirmations(manifest) is True


def test_ask_confirmations_mixed_task_and_subagent() -> None:
    # No confirmation on task tools, but yes on subagent tools
    task1 = _make_task(name="task1")
    task1.tools = []
    task1.subagents = []
    confirmation = MagicMock()
    confirmation.type = "any"
    tool = _make_tool_spec("web", confirmations=[confirmation])
    subagent = _make_subagent_spec("sub1", tools=[tool])
    task2 = _make_task(name="task2")
    task2.tools = []
    task2.subagents = [subagent]
    manifest = _make_manifest(tasks=[task1, task2])
    assert ask_confirmations(manifest) is True


def test_ask_confirmations_empty_confirmations_list_still_false() -> None:
    tool = _make_tool_spec("web", confirmations=[])
    task = _make_task(name="task1")
    task.tools = [tool]
    task.subagents = []
    manifest = _make_manifest(tasks=[task])
    assert ask_confirmations(manifest) is False


def test_ask_confirmations_subagent_no_tools() -> None:
    subagent = _make_subagent_spec("sub1", tools=[])
    task = _make_task(name="task1")
    task.tools = []
    task.subagents = [subagent]
    manifest = _make_manifest(tasks=[task])
    assert ask_confirmations(manifest) is False


# ---------------------------------------------------------------------------
# can_parallelize tests
# ---------------------------------------------------------------------------


def test_can_parallelize_parallelism_none() -> None:
    """Returns False when parallelism is None (not set)."""
    manifest = _make_manifest()
    manifest.meta.parallelism = None
    assert can_parallelize(manifest) is False


def test_can_parallelize_parallelism_1() -> None:
    """Returns False when parallelism is 1."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 1
    assert can_parallelize(manifest) is False


def test_can_parallelize_parallelism_0() -> None:
    """Returns False when parallelism is 0."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 0
    assert can_parallelize(manifest) is False


def test_can_parallelize_parallelism_2_no_tasks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when parallelism > 1 but there are no tasks."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    manifest.tasks = []
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_parallelism_2_task_blocks_parallel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when parallelism > 1 but task has allow_parallel_run=False and no parallel actions."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="blocking", allow_parallel_run=False)
    task.actions = []
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_single_task_allows_parallel_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False with single task that allows parallel run (needs 2 consecutive)."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="parallel_task", allow_parallel_run=True)
    task.actions = []
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_two_consecutive_tasks_allow_parallel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns True when 2 consecutive tasks allow parallel run."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=True)
    t1.actions = []
    t2 = _make_task(name="task2", allow_parallel_run=True)
    t2.actions = []
    manifest.tasks = [t1, t2]
    assert can_parallelize(manifest) is True
    assert "Cannot parallelize" not in caplog.text


def test_can_parallelize_three_consecutive_tasks_allow_parallel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns True when 3 consecutive tasks allow parallel run."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=True)
    t1.actions = []
    t2 = _make_task(name="task2", allow_parallel_run=True)
    t2.actions = []
    t3 = _make_task(name="task3", allow_parallel_run=True)
    t3.actions = []
    manifest.tasks = [t1, t2, t3]
    assert can_parallelize(manifest) is True
    assert "Cannot parallelize" not in caplog.text


def test_can_parallelize_two_parallel_tasks_separated_by_blocking(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when 2 parallel tasks are separated by a blocking task."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=True)
    t1.actions = []
    t2 = _make_task(name="task2", allow_parallel_run=False)
    t2.actions = []
    t3 = _make_task(name="task3", allow_parallel_run=True)
    t3.actions = []
    manifest.tasks = [t1, t2, t3]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_prompts_action_parallel_groups_multiple(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns True with PromptsAction having parallel_message_groups=True and >1 group."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="prompts_task", allow_parallel_run=False)
    action = PromptsAction(
        message_groups=[["hello"], ["world"]], parallel_message_groups=True
    )
    task.actions = [action]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is True
    assert "Cannot parallelize" not in caplog.text


def test_can_parallelize_prompts_action_parallel_groups_single_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False with PromptsAction having parallel_message_groups=True but only 1 group."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="prompts_task", allow_parallel_run=False)
    action = PromptsAction(message_groups=[["hello"]], parallel_message_groups=True)
    task.actions = [action]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_prompts_action_no_parallel_groups(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when PromptsAction has parallel_message_groups=False."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="prompts_task", allow_parallel_run=False)
    action = PromptsAction(
        message_groups=[["hello"], ["world"]], parallel_message_groups=False
    )
    task.actions = [action]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_iterator_action_parallel_groups_multiple(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns True with IteratorAction having parallel_message_groups=True and >1 group."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="iterator_task", allow_parallel_run=False)
    action = IteratorAction(
        input_prompt="test",
        prompt_groups=[["hello"], ["world"]],
        parallel_message_groups=True,
    )
    task.actions = [action]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is True
    assert "Cannot parallelize" not in caplog.text


def test_can_parallelize_iterator_action_parallel_groups_single_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns True with IteratorAction having parallel_message_groups=True even with 1 group."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="iterator_task", allow_parallel_run=False)
    action = IteratorAction(
        input_prompt="test", prompt_groups=[["hello"]], parallel_message_groups=True
    )
    task.actions = [action]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is True
    assert "Cannot parallelize" not in caplog.text


def test_can_parallelize_iterator_action_no_parallel_groups(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when IteratorAction has parallel_message_groups=False."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="iterator_task", allow_parallel_run=False)
    action = IteratorAction(
        input_prompt="test",
        prompt_groups=[["hello"], ["world"]],
        parallel_message_groups=False,
    )
    task.actions = [action]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_cmd_action_only(caplog: pytest.LogCaptureFixture) -> None:
    """Returns False with only CmdAction which has no parallelism support."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="cmd_task", allow_parallel_run=False)
    task.actions = [CmdAction(command="sync")]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_repeater_action(caplog: pytest.LogCaptureFixture) -> None:
    """Returns False with only RepeaterAction which has no parallel_message_groups."""
    from tiz.manifest_parser import RepeaterAction

    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="repeater_task", allow_parallel_run=False)
    task.actions = [RepeaterAction(repeat=2, prompt_groups=[["hello"], ["world"]])]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_scoring_action(caplog: pytest.LogCaptureFixture) -> None:
    """Returns False with only ScoringAction which has no parallel_message_groups."""
    from tiz.manifest_parser import ScoringAction

    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="scoring_task", allow_parallel_run=False)
    task.actions = [
        ScoringAction(
            scoring_prompt="rate", rounds=1, scoring_rounds=1, prompt_groups=[["hello"]]
        )
    ]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_multiple_tasks_one_parallel_not_consecutive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when only one task allows parallelism (not consecutive)."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=False)
    t1.actions = []
    t2 = _make_task(name="task2", allow_parallel_run=True)
    t2.actions = []
    manifest.tasks = [t1, t2]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_all_tasks_block_parallel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when all tasks block parallel execution and have no parallel actions."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=False)
    t1.actions = []
    t2 = _make_task(name="task2", allow_parallel_run=False)
    t2.actions = []
    manifest.tasks = [t1, t2]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_negative_parallelism() -> None:
    """Returns False when parallelism is negative."""
    manifest = _make_manifest()
    manifest.meta.parallelism = -1
    assert can_parallelize(manifest) is False


def test_can_parallelize_action_parallel_groups_plus_consecutive_tasks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns True when action has parallel groups >1 and there are consecutive tasks too."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=True)
    t1.actions = []
    t2 = _make_task(name="task2", allow_parallel_run=True)
    t2.actions = [
        PromptsAction(message_groups=[["a"], ["b"]], parallel_message_groups=True)
    ]
    manifest.tasks = [t1, t2]
    assert can_parallelize(manifest) is True
    assert "Cannot parallelize" not in caplog.text


def test_can_parallelize_all_blocking_tasks_no_parallel_actions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when all tasks block and no action has parallel groups."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=False)
    t1.actions = [CmdAction(command="sync")]
    t2 = _make_task(name="task2", allow_parallel_run=False)
    t2.actions = [
        PromptsAction(message_groups=[["a"], ["b"]], parallel_message_groups=False)
    ]
    manifest.tasks = [t1, t2]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_two_parallel_tasks_no_actions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns True when 2 consecutive tasks allow parallel with no actions."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=True)
    t1.actions = []
    t2 = _make_task(name="task2", allow_parallel_run=True)
    t2.actions = []
    manifest.tasks = [t1, t2]
    assert can_parallelize(manifest) is True
    assert "Cannot parallelize" not in caplog.text


def test_can_parallelize_prompts_action_multiple_groups_no_parallel_flag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when PromptsAction has multiple groups but parallel flag is False."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    task = _make_task(name="prompts_task", allow_parallel_run=False)
    action = PromptsAction(
        message_groups=[["hello"], ["world"]], parallel_message_groups=False
    )
    task.actions = [action]
    manifest.tasks = [task]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text


def test_can_parallelize_first_task_parallel_second_blocking(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Returns False when first task allows parallel but second blocks."""
    manifest = _make_manifest()
    manifest.meta.parallelism = 2
    t1 = _make_task(name="task1", allow_parallel_run=True)
    t1.actions = []
    t2 = _make_task(name="task2", allow_parallel_run=False)
    t2.actions = []
    manifest.tasks = [t1, t2]
    assert can_parallelize(manifest) is False
    assert "no consecutive parallel tasks" in caplog.text
