"""
Batch render point-cloud affordance targets with the IAGNet/Mitsuba style.

This script reads the same render manifest used by utils/base_dataset.py, but
only consumes the independent "point_clouds" list. It writes Mitsuba XML scenes
that follow IAGNet's rend_point.py: normalized points, red/gray affordance
colors, sphere primitives, a rough ground plane, perspective camera, and a large
area light for soft shadows.
"""
import argparse
import importlib
import os
import subprocess
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.base_dataset import (  # noqa: E402
    Modality,
    _load_render_manifest,
    _load_render_point_target,
    _prepare_point_cloud_render_data,
)


XML_HEAD = """<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{camera_origin}" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="{fov}"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="{sample_count}"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="{width}"/>
            <integer name="height" value="{height}"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>
"""


XML_SPHERE = """
    <shape type="sphere">
        <float name="radius" value="{radius}"/>
        <transform name="toWorld">
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
"""


XML_TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="{radiance},{radiance},{radiance}"/>
        </emitter>
    </shape>
</scene>
"""


def _safe_name(value: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))
    return safe.strip('_') or "point_cloud"


def _rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _apply_iagnet_pose(points: np.ndarray) -> np.ndarray:
    """Mirror IAGNet rend_point.py pose: translate, rotate, then remap axes."""
    posed = points.astype(np.float32, copy=True)
    posed += np.array([0.0, 0.2, -0.3], dtype=np.float32)
    posed = posed @ _rotation_matrix_xyz(np.pi / 6.0, 0.0, np.pi / 3.0).T
    return np.stack([posed[:, 2], posed[:, 0], posed[:, 1]], axis=1)


def _write_iagnet_xml(
    xml_path: str,
    points: np.ndarray,
    colors: np.ndarray,
    cfg: Dict,
) -> None:
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    width = int(cfg.get("width", cfg.get("size", 800)))
    height = int(cfg.get("height", cfg.get("size", 800)))
    sphere_radius = float(cfg.get("sphere_radius", 0.025))
    sample_count = int(cfg.get("sample_count", 256))
    fov = float(cfg.get("fov", 25))
    radiance = float(cfg.get("radiance", 6))
    camera_origin = cfg.get("camera_origin", "3,3,3")

    xml_parts = [
        XML_HEAD.format(
            width=width,
            height=height,
            sample_count=sample_count,
            fov=fov,
            camera_origin=camera_origin,
        )
    ]
    posed_points = _apply_iagnet_pose(points)
    for point, color in zip(posed_points, colors):
        xml_parts.append(
            XML_SPHERE.format(
                radius=sphere_radius,
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
                r=float(color[0]),
                g=float(color[1]),
                b=float(color[2]),
            )
        )
    xml_parts.append(XML_TAIL.format(radiance=radiance))

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(''.join(xml_parts))


def _convert_exr_to_jpg(exr_path: str, jpg_path: str) -> None:
    try:
        Imath = importlib.import_module("Imath")
        OpenEXR = importlib.import_module("OpenEXR")
    except ImportError as exc:
        raise RuntimeError("EXR 转 JPG 需要安装 OpenEXR 和 Imath") from exc

    exr_file = OpenEXR.InputFile(exr_path)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    data_window = exr_file.header()['dataWindow']
    size = (
        data_window.max.x - data_window.min.x + 1,
        data_window.max.y - data_window.min.y + 1,
    )
    rgb = [np.frombuffer(exr_file.channel(c, pixel_type), dtype=np.float32) for c in 'RGB']
    rgb8 = []
    for channel in rgb:
        channel = np.where(
            channel <= 0.0031308,
            channel * 12.92,
            1.055 * np.power(np.maximum(channel, 0.0), 1.0 / 2.4) - 0.055,
        )
        channel = np.clip(channel * 255.0, 0, 255).astype(np.uint8)
        rgb8.append(Image.frombytes("L", size, channel.tobytes()))
    os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
    Image.merge("RGB", rgb8).save(jpg_path, "JPEG", quality=95)


def _run_mitsuba_cli(xml_path: str, mitsuba_bin: str) -> Optional[str]:
    result = subprocess.run(
        [mitsuba_bin, os.path.basename(xml_path)],
        cwd=os.path.dirname(xml_path),
        check=False,
    )
    if result.returncode != 0:
        warnings.warn(f"Mitsuba 渲染失败: {xml_path}")
        return None
    exr_path = os.path.splitext(xml_path)[0] + ".exr"
    return exr_path if os.path.exists(exr_path) else None


def _init_mitsuba_python(variant: str):
    """Import Mitsuba with a non-LLVM scalar variant to avoid Dr.Jit LLVM errors."""
    os.environ.setdefault("MITSUBA_VARIANT", variant)
    mi = importlib.import_module("mitsuba")

    try:
        mi.set_variant(variant)
    except Exception as exc:
        current = None
        try:
            current = mi.variant()
        except Exception:
            pass
        if current != variant:
            raise RuntimeError(
                f"无法设置 Mitsuba variant={variant}。如果你遇到 LLVM 报错，"
                "请确认没有在别处提前设置 llvm_* / cuda_* variant。"
            ) from exc
    return mi


def _parse_vec3(text: str) -> List[float]:
    if isinstance(text, (list, tuple)):
        return [float(v) for v in text]
    return [float(v.strip()) for v in str(text).split(',')]


def _mitsuba_rgb(value: Tuple[float, float, float]) -> Dict:
    return {"type": "rgb", "value": [float(value[0]), float(value[1]), float(value[2])]}


def _build_mitsuba_scene_dict(points: np.ndarray, colors: np.ndarray, cfg: Dict, mi) -> Dict:
    width = int(cfg.get("width", cfg.get("size", 800)))
    height = int(cfg.get("height", cfg.get("size", 800)))
    sphere_radius = float(cfg.get("sphere_radius", 0.025))
    sample_count = int(cfg.get("sample_count", 256))
    fov = float(cfg.get("fov", 25))
    radiance = float(cfg.get("radiance", 6))
    camera_origin = _parse_vec3(cfg.get("camera_origin", "3,3,3"))

    transform = mi.ScalarTransform4f
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": -1},
        "sensor": {
            "type": "perspective",
            "near_clip": 0.1,
            "far_clip": 100.0,
            "fov": fov,
            "to_world": transform.look_at(
                origin=camera_origin,
                target=[0.0, 0.0, 0.0],
                up=[0.0, 0.0, 1.0],
            ),
            "sampler": {
                "type": "independent",
                "sample_count": sample_count,
            },
            "film": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
                "pixel_format": "rgb",
                "rfilter": {"type": "gaussian"},
            },
        },
        "ground": {
            "type": "rectangle",
            "to_world": transform.translate([0.0, 0.0, -0.5]) @ transform.scale([10.0, 10.0, 1.0]),
            "bsdf": {
                "type": "roughplastic",
                "distribution": "ggx",
                "alpha": 0.05,
                "int_ior": 1.46,
                "diffuse_reflectance": _mitsuba_rgb((1.0, 1.0, 1.0)),
            },
        },
        "area_light": {
            "type": "rectangle",
            "to_world": transform.look_at(
                origin=[-4.0, 4.0, 20.0],
                target=[0.0, 0.0, 0.0],
                up=[0.0, 0.0, 1.0],
            ) @ transform.scale([10.0, 10.0, 1.0]),
            "emitter": {
                "type": "area",
                "radiance": _mitsuba_rgb((radiance, radiance, radiance)),
            },
        },
    }

    posed_points = _apply_iagnet_pose(points)
    for idx, (point, color) in enumerate(zip(posed_points, colors)):
        scene[f"sphere_{idx:05d}"] = {
            "type": "sphere",
            "center": [float(point[0]), float(point[1]), float(point[2])],
            "radius": sphere_radius,
            "bsdf": {
                "type": "diffuse",
                "reflectance": _mitsuba_rgb((float(color[0]), float(color[1]), float(color[2]))),
            },
        }
    return scene


def _run_mitsuba_python(
    points: np.ndarray,
    colors: np.ndarray,
    exr_path: str,
    cfg: Dict,
    variant: str = "scalar_rgb",
) -> str:
    mi = _init_mitsuba_python(variant)
    scene_dict = _build_mitsuba_scene_dict(points, colors, cfg, mi)
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene)
    os.makedirs(os.path.dirname(exr_path), exist_ok=True)
    mi.util.write_bitmap(exr_path, image)
    return exr_path


def export_iagnet_style(
    manifest_path: str,
    dataset_root_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    run_mitsuba: bool = False,
    convert_jpg: bool = False,
    mitsuba_bin: str = "mitsuba",
    mitsuba_renderer: str = "python",
    mitsuba_variant: str = "scalar_rgb",
) -> Dict[str, List[str]]:
    manifest = _load_render_manifest(manifest_path, dataset_root_override=dataset_root_override)
    dataset_root = manifest["dataset_root"]
    output_dir = output_dir_override or manifest["output_dir"]
    iagnet_cfg = manifest.get("render", {}).get("point_cloud", {}).get("iagnet", {})
    pc_cfg = manifest.get("render", {}).get("point_cloud", {})
    xml_cfg = {**pc_cfg, **iagnet_cfg}
    max_points = xml_cfg.get("max_points", 2048)

    xml_dir = os.path.join(output_dir, "iagnet_xml")
    exr_dir = os.path.join(output_dir, "iagnet_exr")
    jpg_dir = os.path.join(output_dir, "iagnet_jpg")
    point_items = manifest.get("point_clouds", [])
    if not isinstance(point_items, list) or not point_items:
        raise ValueError("manifest 需要提供非空 point_clouds 数组")

    outputs = {"xml": [], "exr": [], "jpg": []}
    for idx, item in enumerate(point_items):
        obj_type = item.get("obj_type")
        pc_id = item.get("pc_id")
        aff = item.get("aff")
        if obj_type is None or pc_id is None or aff is None:
            warnings.warn(f"跳过 point_clouds[{idx}]：缺少 obj_type、pc_id 或 aff")
            continue

        points, mask = _load_render_point_target(dataset_root, obj_type, int(pc_id), aff)
        points, colors = _prepare_point_cloud_render_data(points, mask, max_points=max_points)
        name = item.get("name") or f"{Modality._normalize_label(obj_type)}_pc{pc_id}_{Modality._normalize_label(aff)}"
        xml_path = os.path.join(xml_dir, f"{_safe_name(name)}.xml")
        _write_iagnet_xml(xml_path, points, colors, xml_cfg)
        outputs["xml"].append(xml_path)
        print(f"XML saved: {xml_path}")

        if run_mitsuba:
            if mitsuba_renderer == "python":
                exr_path = os.path.join(exr_dir, f"{_safe_name(name)}.exr")
                try:
                    exr_path = _run_mitsuba_python(
                        points,
                        colors,
                        exr_path,
                        xml_cfg,
                        variant=mitsuba_variant,
                    )
                except Exception as exc:
                    warnings.warn(
                        "Mitsuba Python API 渲染失败。若报 LLVM 相关错误，请确认使用 "
                        f"--mitsuba-variant scalar_rgb，且没有提前导入 mitsuba 并设置 llvm/cuda variant。错误: {exc}"
                    )
                    exr_path = None
            else:
                exr_path = _run_mitsuba_cli(xml_path, mitsuba_bin)
            if exr_path is not None:
                outputs["exr"].append(exr_path)
                print(f"EXR saved: {exr_path}")
                if convert_jpg:
                    jpg_path = os.path.join(jpg_dir, f"{_safe_name(name)}.jpg")
                    _convert_exr_to_jpg(exr_path, jpg_path)
                    outputs["jpg"].append(jpg_path)
                    print(f"JPG saved: {jpg_path}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IAGNet/Mitsuba-style point-cloud render scenes.")
    parser.add_argument("--render-json", required=True, help="Render manifest JSON path.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset_root in manifest.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir in manifest.")
    parser.add_argument("--run-mitsuba", action="store_true", help="Run mitsuba for each generated XML.")
    parser.add_argument(
        "--mitsuba-renderer",
        choices=("python", "cli"),
        default="python",
        help="Use Mitsuba Python API by default; cli keeps the original executable workflow.",
    )
    parser.add_argument(
        "--mitsuba-variant",
        default="scalar_rgb",
        help="Mitsuba Python variant. scalar_rgb avoids Dr.Jit LLVM/JIT dependency.",
    )
    parser.add_argument("--mitsuba-bin", default="mitsuba", help="Mitsuba executable name or path.")
    parser.add_argument("--convert-jpg", action="store_true", help="Convert rendered EXR files to JPG.")
    args = parser.parse_args()

    outputs = export_iagnet_style(
        args.render_json,
        dataset_root_override=args.dataset_root,
        output_dir_override=args.output_dir,
        run_mitsuba=args.run_mitsuba,
        convert_jpg=args.convert_jpg,
        mitsuba_bin=args.mitsuba_bin,
        mitsuba_renderer=args.mitsuba_renderer,
        mitsuba_variant=args.mitsuba_variant,
    )
    print("Done.")
    for group, paths in outputs.items():
        print(f"{group}: {len(paths)}")


if __name__ == "__main__":
    main()
