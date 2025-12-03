from __future__ import annotations
from dataclasses import dataclass
import json
import os
from typing import Optional

@dataclass
class DataPoint:
    timestamp: float
    x_value: float
    y_value: float

@dataclass
class TimeSeries:
    name: str
    points: list[DataPoint]

    def downsample(
        self, factor: int
    ) -> TimeSeries:
        # Downsamples each time series by the given factor
        if factor <= 1:
            return self
        
        ds_points: list[DataPoint] = []
        for i in range(0, len(self.points), factor):
            ds_points.append(self.points[i])
        self.points = ds_points
        return self
    
    def smooth(
        self, window: int
    ) -> TimeSeries:
        if window <= 1:
            return self

        half: int = window // 2

        smoothed_points: list[DataPoint] = []
        for i in range(len(self.points)):
            start: int = max(0, i - half)
            end: int = min(len(self.points), i + half + 1)
            window_points: list[DataPoint] = self.points[start:end]

            avg_x: float = sum(p.x_value for p in window_points) / len(window_points)
            avg_y: float = sum(p.y_value for p in window_points) / len(window_points)

            smoothed_points.append(
                DataPoint(
                    timestamp=self.points[i].timestamp,
                    x_value=avg_x,
                    y_value=avg_y,
                )
            )
        self.points = smoothed_points
        return self  
    
    def __str__(self) -> str:
        points_str = self._points_to_lilaq_str()

        return (
            f"""
            lq.plot(
                {points_str},
            )
            """
        )

    def _points_to_lilaq_str(self) -> str:
        lines_x: list[str] = ["/* X */", "("]
        lines_y: list[str] = ["/* Y */", "("]

        for point in self.points:
            lines_x.append(f"{point.x_value}, ")
            lines_y.append(f"{point.y_value}, ")
        
        lines_x.append("),")
        lines_y.append("),")

        return "\n".join(lines_x) + "\n" + "\n".join(lines_y)

@dataclass
class LilaqLimit:
    min: Optional[float]
    max: Optional[float]

    def __str__(self) -> str:
        return f"({self.min if self.min is not None else 'auto'}, {self.max if self.max is not None else 'auto'})"


@dataclass
class LilaqDiagram:
    xlim: Optional[LilaqLimit]
    xlabel: str

    ylim: Optional[LilaqLimit]
    ylabel: str

    series: list[TimeSeries]

    def _series_to_lilaq_str(self) -> str:
        return ",\n".join(str(s) for s in self.series)

    def __str__(self) -> str:
        return f"""
            #import "@preview/lilaq:0.5.0" as lq
            #figure(
                lq.diagram(
                    xlim: {self.xlim},
                    xlabel: [{self.xlabel}],

                    ylim: {self.ylim},
                    ylabel: [{self.ylabel}],

                    {self._series_to_lilaq_str()}
                )
            )
            """
    
    def downsample(
        self, factor: int
    ) -> LilaqDiagram:
        for s in self.series:
            s.downsample(factor)
        return self
    
    def smooth(
        self, window: int
    ) -> LilaqDiagram:
        for s in self.series:
            s.smooth(window)
        return self

def load_series(
    directory: str, experiments: list[str], metric: str
) -> list[TimeSeries]:
    data: list[TimeSeries] = []

    for exp in experiments:
        filename: str = f"{exp}.{metric}.json"
        path: str = os.path.join(directory, filename)

        if not os.path.isfile(path):
            continue

        time_series = TimeSeries(name=exp.capitalize(), points=[])
        with open(path, "r", encoding="utf-8") as f:
            raw: list[list[float]] = json.load(f)
            time_series.points.extend(
                DataPoint(entry[0], entry[1], entry[2]) for entry in raw
            )
            data.append(time_series)

    return data

def generate_lilaq_string(
    directory: str,
    experiments: list[str],
    metric: str,
    resolution: int = 1,
    smooth_window: int = 1,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> str:
    
    return LilaqDiagram(
        xlim=LilaqLimit(min=x_min, max=x_max),
        xlabel="Episodes",
        ylim=LilaqLimit(min=y_min, max=y_max),
        ylabel=metric.capitalize(),
        series=load_series(directory, experiments, metric),
    ).downsample(resolution).smooth(smooth_window).__str__()

    
