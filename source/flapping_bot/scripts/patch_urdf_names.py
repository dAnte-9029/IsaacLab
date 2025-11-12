"""批量修改 URDF 中的关节命名并可选转换为 USD。

用法示例（推荐用 isaaclab.bat 调起）：

  仅改名保存新 URDF：
    .\isaaclab.bat -p source/flapping_bot/scripts/patch_urdf_names.py \
      --src "F:/path/to/robot.urdf" --dst-name flap_robot_v50

  改名后同时生成 USD：
    .\isaaclab.bat -p source/flapping_bot/scripts/patch_urdf_names.py \
      --src "F:/path/to/robot.urdf" --dst-name flap_robot_v50 --to-usd

  自定义映射：
    .\isaaclab.bat -p source/flapping_bot/scripts/patch_urdf_names.py \
      --src ".../robot.urdf" --map youweijoint=right_tail youyijoint=right_wing

默认映射包含：youweijoint->right_tail，youyijoint->right_wing。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


def parse_kv_list(pairs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"无效映射 '{item}'，应为 old=new 形式")
        old, new = item.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise argparse.ArgumentTypeError(f"无效映射 '{item}'，键值不能为空")
        mapping[old] = new
    return mapping


def patch_urdf_names(src: Path, dst: Path, name_map: dict[str, str]) -> None:
    tree = ET.parse(src)
    root = tree.getroot()

    # helper: tag 后缀匹配（忽略命名空间）
    def tag_endswith(elem: ET.Element, suffix: str) -> bool:
        t = elem.tag
        if "}" in t:
            t = t.split("}", 1)[1]
        return t.lower().endswith(suffix)

    # 遍历所有元素，处理：
    # - <joint name="old"> -> name="new"（仅 joint 元素）
    # - mimic 等引用：任意元素的属性 joint="old" -> joint="new"
    # - transmission/joint 子元素：<joint name="old"> -> name="new"
    count_joint_renamed = 0
    count_refs_updated = 0

    for elem in root.iter():
        # 关节自身改名
        if tag_endswith(elem, "joint") and "name" in elem.attrib:
            old = elem.attrib["name"]
            if old in name_map:
                elem.set("name", name_map[old])
                count_joint_renamed += 1

        # mimic 或其它引用 joint="..."
        if "joint" in elem.attrib and elem.attrib["joint"] in name_map:
            elem.set("joint", name_map[elem.attrib["joint"]])
            count_refs_updated += 1

        # transmission 下的 joint 子元素
        if tag_endswith(elem, "transmission"):
            for ch in elem:
                if tag_endswith(ch, "joint") and "name" in ch.attrib:
                    old = ch.attrib["name"]
                    if old in name_map:
                        ch.set("name", name_map[old])
                        count_refs_updated += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"[INFO] 已写出 URDF: {dst}")
    print(f"[INFO] joint 重命名: {count_joint_renamed} 处, 引用更新: {count_refs_updated} 处")


def convert_to_usd(urdf_path: Path, usd_out: Path) -> None:
    # 使用 AppLauncher 确保在导入 omni/isaacsim 前初始化
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.converters import UrdfConverter

        cfg = sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            usd_dir=str(usd_out.parent),
            usd_file_name=str(usd_out.name),
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            make_instanceable=False,
            force_usd_conversion=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
        )
        converter = UrdfConverter(cfg)
        print(f"[INFO] 已生成 USD: {converter.usd_path}")
    finally:
        app.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="重命名 URDF 关节并可选生成 USD")
    ap.add_argument("--src", type=str, default=str(SRC_ROOT / "isaaclab_assets/data/flapping_bot/robots/robot.urdf"), help="源 URDF 路径")
    ap.add_argument("--dst-name", type=str, default="flap_robot_v50", help="目标 URDF/ USD 基础文件名（不含后缀）")
    ap.add_argument("--dst-dir", type=str, default=str(SRC_ROOT / "isaaclab_assets/data/flapping_bot/robots"), help="输出目录（URDF 与 USD 保存位置）")
    ap.add_argument("--map", nargs="*", default=["youweijoint=right_tail", "youyijoint=right_wing"], help="关节名映射，形如 old=new，可多对")
    ap.add_argument("--to-usd", action="store_true", help="完成改名后同时生成 USD")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[ERR] 源 URDF 不存在: {src}")
        sys.exit(2)

    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_urdf = dst_dir / f"{args.dst_name}.urdf"
    dst_usd = dst_dir / f"{args.dst_name}.usd"

    mapping = parse_kv_list(args.map)
    print(f"[INFO] 映射表: {mapping}")
    print(f"[INFO] 源 URDF: {src}")
    print(f"[INFO] 目标 URDF: {dst_urdf}")

    patch_urdf_names(src, dst_urdf, mapping)

    if args.to_usd:
        print(f"[INFO] 开始转换 USD -> {dst_usd}")
        convert_to_usd(dst_urdf, dst_usd)


if __name__ == "__main__":
    main()

