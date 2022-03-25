import torch
import random
import pydiffvg
from typing import List, Tuple


def initialize_curves(num_paths, canvas_width, canvas_height, black_and_white=False) -> Tuple[List[pydiffvg.Path], List[pydiffvg.ShapeGroup]]:
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points=num_control_points, points=points, stroke_width=torch.tensor(1.0),
                             is_closed=False)
        shapes.append(path)
        if black_and_white:
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                             stroke_color=torch.tensor([0., 0., 0., 1.])
                                             )
        else:
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                             stroke_color=torch.tensor(
                                                 [random.random(), random.random(), random.random(), random.random()]))
        shape_groups.append(path_group)
    return shapes, shape_groups


def initialize_circles(num_points, canvas_width, canvas_height, radius=3., black_and_white=False) -> Tuple[List[pydiffvg.Circle], List[pydiffvg.ShapeGroup]]:
    shapes = []
    shape_groups = []
    for i in range(num_points):
        point = [random.random() * canvas_width, random.random() * canvas_height]
        circle = pydiffvg.Circle(radius=torch.tensor(radius), center=torch.tensor(point))
        shapes.append(circle)

        if black_and_white:
            circle_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0., 0., 0., 1.]))
        else:
            circle_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([0]),
                fill_color=torch.tensor([random.random(), random.random(), random.random(), random.random()])
            )
        shape_groups.append(circle_group)
    return shapes, shape_groups
