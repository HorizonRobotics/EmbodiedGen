import pytest
from huggingface_hub import snapshot_download
from embodied_gen.data.asset_converter import AssetConverterFactory, AssetType


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("EmbodiedGenData")
    snapshot_download(
        repo_id="HorizonRobotics/EmbodiedGenData",
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns="demo_assets/remote_control/*",
        local_dir_use_symlinks=False,
    )
    return data_dir


def test_MeshtoMJCFConverter(data_dir):
    urdf_path = (
        data_dir / "demo_assets/remote_control/result/remote_control.urdf"
    )
    assert urdf_path.exists(), f"URDF not found: {urdf_path}"

    output_file = (
        data_dir / "demo_assets/remote_control/mjcf/remote_control.xml"
    )
    asset_converter = AssetConverterFactory.create(
        target_type=AssetType.MJCF,
        source_type=AssetType.URDF,
    )

    with asset_converter:
        asset_converter.convert(str(urdf_path), str(output_file))

    assert output_file.exists(), f"Output not generated: {output_file}"
    assert output_file.stat().st_size > 0


def test_MeshtoUSDConverter(data_dir):
    urdf_path = (
        data_dir / "demo_assets/remote_control/result/remote_control.urdf"
    )
    assert urdf_path.exists(), f"URDF not found: {urdf_path}"

    output_file = (
        data_dir / "demo_assets/remote_control/usd/remote_control.usd"
    )
    asset_converter = AssetConverterFactory.create(
        target_type=AssetType.USD,
        source_type=AssetType.MESH,
    )

    with asset_converter:
        asset_converter.convert(str(urdf_path), str(output_file))

    assert output_file.exists(), f"Output not generated: {output_file}"
    assert output_file.stat().st_size > 0


from typing import Literal


def test_cvt_embodiedgen_asset_to_anysim(
    simulator_name: Literal[
        "isaacsim",
        "isaacgym",
        "genesis",
        "pybullet",
        "sapien3",
        "mujoco",
    ] = "mujoco",
):
    from embodied_gen.data.asset_converter import (
        SimAssetMapper,
        cvt_embodiedgen_asset_to_anysim,
    )

    dst_asset_path = cvt_embodiedgen_asset_to_anysim(
        urdf_files=[
            "outputs/embodiedgen_assets/demo_assets/remote_control2/result/remote_control.urdf",
        ],
        target_type=SimAssetMapper[simulator_name],
        source_type=AssetType.MESH,
        overwrite=True,
    )
