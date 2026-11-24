from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH"),
    reason="native ComfyUI source fixture is not configured",
)


def _comfyui_root() -> Path:
    return Path(os.environ["COMFYUI_PATH"]).resolve()


def _source_tree(relative_path: str) -> ast.Module:
    path = _comfyui_root() / relative_path
    assert path.is_file(), f"reviewed ComfyUI source is missing {relative_path}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"reviewed ComfyUI source is missing class {name}")


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"reviewed ComfyUI source is missing function {name}")


def _argument_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = function.args
    return {
        *(item.arg for item in args.posonlyargs),
        *(item.arg for item in args.args),
        *(item.arg for item in args.kwonlyargs),
    }


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def test_custom_node_imports_against_reviewed_comfyui_source():
    import folder_paths
    import nodes

    key = "MinimaxH3LatentUpscaler3DRefineHandoff"
    assert key in nodes.NODE_CLASS_MAPPINGS
    node_cls = nodes.NODE_CLASS_MAPPINGS[key]
    assert node_cls.INPUT_IS_LIST is True
    assert node_cls.OUTPUT_IS_LIST == (True,)
    assert node_cls.RETURN_TYPES == ("LATENT",)

    schema = node_cls.INPUT_TYPES()
    assert {"latent", "noise", "sampler", "sigmas"}.issubset(schema["required"])
    assert {"audio_latent", "refine_state", "model", "positive", "negative"}.issubset(
        schema["optional"]
    )
    assert "latent_upscale_models" in folder_paths.folder_names_and_paths


def test_refinement_runtime_matches_reviewed_comfyui_source_contract():
    samplers_tree = _source_tree("comfy/samplers.py")
    guider = _class(samplers_tree, "CFGGuider")
    methods = {
        node.name: node
        for node in guider.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"inner_set_conds", "set_conds", "set_cfg", "sample"}.issubset(methods)
    assert {
        "noise",
        "latent_image",
        "sampler",
        "sigmas",
        "denoise_mask",
        "callback",
        "disable_pbar",
        "seed",
    }.issubset(_argument_names(methods["sample"]))

    assert _function(_source_tree("comfy/sample.py"), "fix_empty_latent_channels")
    assert _function(_source_tree("comfy/model_management.py"), "intermediate_device")
    assert _function(_source_tree("latent_preview.py"), "prepare_callback")
    assert _class(_source_tree("comfy/nested_tensor.py"), "NestedTensor")
    assert "PROGRESS_BAR_ENABLED" in _assigned_names(_source_tree("comfy/utils.py"))


@pytest.mark.parametrize("kind", ["2d", "3d"])
def test_upscaler_external_model_paths(tmp_path, monkeypatch, kind):
    import importlib.util
    import sys

    import folder_paths
    import torch
    from safetensors.torch import save_file
    from utils.extra_config import load_extra_path_config

    import nodes

    category = "latent_upscale_models"
    model_root = tmp_path / "models"
    standard = model_root / category
    external = tmp_path / "external"
    standard.mkdir(parents=True)
    external.mkdir()
    (external / "nested").mkdir()
    (standard / "local.pth").touch()
    (standard / "duplicate.pth").touch()
    (external / "duplicate.pth").touch()
    (external / "ignore.txt").touch()
    checkpoint = external / "nested" / "external.SAFETENSORS"
    save_file({"upscaler.weight": torch.ones(2)}, str(checkpoint))

    monkeypatch.setattr(folder_paths, "models_dir", str(model_root))
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {})
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    config = tmp_path / "extra_model_paths.yaml"
    config.write_text(
        "external:\n  latent_upscale_models: external\n", encoding="utf-8"
    )
    load_extra_path_config(str(config))

    # Import a fresh copy without mutating the already registered node modules
    # or their model caches. This exercises registration after YAML loading.
    name = f"nodes._extra_paths_test_{kind}"
    spec = importlib.util.spec_from_file_location(
        name, Path(nodes.__file__).parent / f"minimax_h3_latent_upscaler_{kind}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    assert folder_paths.get_folder_paths(category) == [str(external), str(standard)]
    nested_name = os.path.join("nested", checkpoint.name)
    assert module.scan_models() == sorted(["local.pth", "duplicate.pth", nested_name])
    state = module._load_raw_sd(str(checkpoint), torch.device("cpu"), torch.float32)
    assert list(state) == ["weight"]
    assert torch.equal(state["weight"], torch.ones(2))

    # Stop at the checkpoint I/O boundary: verify load_model actually resolves
    # the selected relative name, not just that the combo lists it.
    class ResolvedPath(Exception):
        pass

    def inspect_path(path, _device, _dtype):
        raise ResolvedPath(path)

    monkeypatch.setattr(module, "_load_raw_sd", inspect_path)
    for filename, expected in (
        (nested_name, checkpoint),
        ("duplicate.pth", external / "duplicate.pth"),
        ("local.pth", standard / "local.pth"),
    ):
        with pytest.raises(ResolvedPath) as exc:
            module.load_model(filename, torch.device("cpu"), "fp32")
        assert Path(exc.value.args[0]) == expected
    with pytest.raises(FileNotFoundError):
        module.load_model("missing.pth", torch.device("cpu"), "fp32")
