import math
import random
import numpy as np
from PIL import Image, ImageDraw


def random_bbox(img_h, img_w, mask_h, mask_w, margin_h, margin_w, max_delta_h, max_delta_w):
    maxt = img_h - margin_h - mask_h
    maxl = img_w - margin_w - mask_w
    bbox_list = []
    t = random.randint(margin_h, maxt)
    l = random.randint(margin_w, maxl)
    bbox_list.append((t, l, img_h, img_w))
    delta_h = random.randint(0, max_delta_h // 2 + 1)
    delta_w = random.randint(0, max_delta_w // 2 + 1)
    return t + delta_h, t + img_h - delta_h, l + delta_w, l + img_w - delta_w


def brush_stroke_mask(H, W):
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 12
    max_width = 40

    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2,
                          v[1] - width // 2,
                          v[0] + width // 2,
                          v[1] + width // 2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, 1, H, W))
    return mask