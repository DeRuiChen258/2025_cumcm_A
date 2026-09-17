import pandas as pd
import pytest

from src.export import RESULT3_COLS, export_result1, export_result2, export_result3


@pytest.fixture
def fake_q3():
    return {
        "shell_rows": [{
            "弹编号": i + 1, "无人机": "FY1", "目标导弹": "M1",
            "速度 v (m/s)": 100.0, "航向角 θ (rad)": 3.14,
            "投放时刻 t_d (s)": 1.0, "起爆时刻 t_e (s)": 5.0,
            "投放点 x": 1.0, "投放点 y": 2.0, "投放点 z": 3.0,
            "起爆点 x": 4.0, "起爆点 y": 5.0, "起爆点 z": 6.0,
            "贡献遮蔽区间 (s)": "[[1, 2]]", "遮蔽时长 (s)": 1.0,
        } for i in range(3)]
    }


def test_export_readback(tmp_path, fake_q3):
    p1 = export_result1(fake_q3, str(tmp_path / "result1.xlsx"))
    df1 = pd.read_excel(p1)
    assert df1.shape == (3, 13)
    p2 = export_result2(fake_q3, str(tmp_path / "result2.xlsx"))
    df2 = pd.read_excel(p2)
    assert df2.shape == (3, 13)
    p3 = export_result3(fake_q3, str(tmp_path / "result3.xlsx"))
    df3 = pd.read_excel(p3)
    assert list(df3.columns) == RESULT3_COLS
    assert df3.shape == (3, 15)
