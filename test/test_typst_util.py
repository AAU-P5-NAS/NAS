import json
import pytest
from src.utils.typst.parser import (
    DataPoint,
    LilaqPlot,
    LilaqLimit,
    LilaqDiagram,
    load_json_into_plots,
    generate_lilaq_string,
    indent,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_points() -> list[DataPoint]:
    return [
        DataPoint(timestamp=0.0, x_value=1.0, y_value=10.0),
        DataPoint(timestamp=1.0, x_value=2.0, y_value=20.0),
        DataPoint(timestamp=2.0, x_value=3.0, y_value=30.0),
        DataPoint(timestamp=3.0, x_value=4.0, y_value=40.0),
        DataPoint(timestamp=4.0, x_value=5.0, y_value=50.0),
    ]


@pytest.fixture
def sample_plot(sample_points) -> LilaqPlot:
    return LilaqPlot(label="Test", points=list(sample_points), every=None)


# -----------------------------------------------------------------------------
# Tests: downsample
# -----------------------------------------------------------------------------

def test_downsample_factor_1(sample_plot):
    sample_plot.downsample(1)
    assert len(sample_plot.points) == 5


def test_downsample_factor_2(sample_plot):
    sample_plot.downsample(2)
    assert [p.x_value for p in sample_plot.points] == [1.0, 3.0, 5.0]


def test_downsample_factor_large(sample_plot):
    sample_plot.downsample(10)
    assert len(sample_plot.points) == 1


# -----------------------------------------------------------------------------
# Tests: smooth
# -----------------------------------------------------------------------------

def test_smooth_window_1(sample_plot):
    sample_plot.smooth(1)
    assert [p.y_value for p in sample_plot.points] == [10, 20, 30, 40, 50]


def test_smooth_window_3(sample_plot):
    sample_plot.smooth(3)
    ys = [p.y_value for p in sample_plot.points]
    expected = [
        (10 + 20) / 2,
        (10 + 20 + 30) / 3,
        (20 + 30 + 40) / 3,
        (30 + 40 + 50) / 3,
        (40 + 50) / 2,
    ]
    assert ys == pytest.approx(expected)


# -----------------------------------------------------------------------------
# Tests: indent()
# -----------------------------------------------------------------------------

def test_indent():
    assert indent(0) == ""
    assert indent(1) == "\t"
    assert indent(3) == "\t\t\t"


# -----------------------------------------------------------------------------
# Tests: _points_to_lilaq_str
# -----------------------------------------------------------------------------

def test_points_to_str(sample_plot):
    s = sample_plot._points_to_lilaq_str(indent_level=0)
    assert "/* X */" in s
    assert "/* Y */" in s
    assert "(1.0," not in s  # indentation separates
    assert "1.0," in s       # value included


# -----------------------------------------------------------------------------
# Tests: LilaqPlot.to_str
# -----------------------------------------------------------------------------

def test_plot_to_str(sample_plot):
    output = sample_plot.to_str(indent_level=0)
    assert "lq.plot(" in output
    assert "label: [Test]" in output
    assert "every: none" in output
    assert "/* X */" in output


# -----------------------------------------------------------------------------
# Tests: LilaqLimit
# -----------------------------------------------------------------------------

def test_lilaq_limit_to_str():
    assert LilaqLimit(0, 10).to_str() == "(0, 10)"
    assert LilaqLimit(None, 10).to_str() == "(auto, 10)"
    assert LilaqLimit(5, None).to_str() == "(5, auto)"


# -----------------------------------------------------------------------------
# Tests: LilaqDiagram
# -----------------------------------------------------------------------------

def test_diagram_has_multiple():
    p1 = LilaqPlot("A", [], None)
    p2 = LilaqPlot("B", [], None)
    d = LilaqDiagram(
        xlim=LilaqLimit(None, None),
        xlabel="Episodes",
        ylim=LilaqLimit(None, None),
        ylabel="Accuracy",
        series=[p1, p2],
    )
    assert d.has_multiple() is True


def test_diagram_single():
    p = LilaqPlot("A", [], None)
    d = LilaqDiagram(
        xlim=LilaqLimit(None, None),
        xlabel="Episodes",
        ylim=LilaqLimit(None, None),
        ylabel="Accuracy",
        series=[p],
    )
    assert d.has_multiple() is False


# -----------------------------------------------------------------------------
# Tests: load_json_into_plots
# -----------------------------------------------------------------------------

def test_load_json_into_plots(tmp_path):
    data = [
        [0, 1, 10],
        [1, 2, 20],
    ]

    directory = tmp_path
    file_path = directory / "exp1.accuracy.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    plots = load_json_into_plots(str(directory), ["exp1"], "accuracy", every=None)

    assert len(plots) == 1
    assert plots[0].label == "Exp1"
    assert len(plots[0].points) == 2
    assert plots[0].points[1].y_value == 20


def test_load_json_skips_missing(tmp_path):
    plots = load_json_into_plots(str(tmp_path), ["missing"], "acc", every=None)
    assert plots == []


# -----------------------------------------------------------------------------
# Tests: generate_lilaq_string
# -----------------------------------------------------------------------------

def test_generate_lilaq_string(tmp_path):
    data = [
        [0, 1, 10],
        [1, 2, 20],
    ]
    directory = tmp_path
    file_path = directory / "exp.accuracy.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    output = generate_lilaq_string(
        directory=str(directory),
        experiments=["exp"],
        metric="accuracy",
        resolution=1,
        smooth_window=1,
    )

    assert "#import" in output
    assert "lq.diagram(" in output
    assert "lq.plot(" in output
    assert "xlabel: [Episodes]" in output
    assert "ylabel: [Accuracy]" in output
    assert "/* X */" in output
    assert "1," in output
    assert "10," in output
    assert "/* Y */" in output
    assert "2," in output
    assert "20," in output