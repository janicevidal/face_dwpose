#!/usr/bin/env python3
"""
Convert RAW to PNG, crop based on person mask's SQUARE bbox area ratio.
If square area / image area >= 50%: keep full image.
Otherwise: expand the square by 20% and crop to that region (preserving original square).
Then resize to max long edge 1280px.
"""

import os
import argparse
import cv2
import numpy as np
import rawpy
import imageio
from glob import glob

RAW_EXTS = ('.NEF', '.CR2', '.ARW', '.RAF', '.RW2', '.DNG', '.ORF', '.PEF')

def get_person_bbox(mask):
    """返回人像最小外接矩形 (x, y, w, h)。若无则返回 None。"""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    return cv2.boundingRect(coords)

def bbox_to_containing_square(bbox, img_shape):
    """
    生成一个正方形 (x, y, side, side)，完全包含给定的矩形 bbox，
    且尽量不超出图像边界。如果无法完全包含（如 side 大于图像尺寸），返回 None。
    """
    x, y, w, h = bbox
    side = max(w, h)
    img_h, img_w = img_shape[:2]

    if side > img_w or side > img_h:
        return None

    # 计算左边界允许的范围 [left_min, left_max]
    left_min = max(0, x + w - side)   # 矩形右边界对齐正方形右边界时的左边界
    left_max = min(x, img_w - side)   # 矩形左边界对齐正方形左边界时的左边界
    if left_min > left_max:
        return None
    # 选择居中位置
    left = (left_min + left_max) // 2

    # 计算上边界允许的范围
    top_min = max(0, y + h - side)
    top_max = min(y, img_h - side)
    if top_min > top_max:
        return None
    top = (top_min + top_max) // 2

    return left, top, side, side

def square_area_ratio(square_bbox, img_shape):
    """计算正方形面积与整图面积的比值。"""
    _, _, side, _ = square_bbox
    img_h, img_w = img_shape[:2]
    sq_area = side * side
    img_area = img_w * img_h
    return sq_area / img_area if img_area > 0 else 0

def expand_and_fit(square_bbox, img_shape, expand_ratio=0.2):
    """
    将正方形向外扩展 expand_ratio，然后平移使其尽可能落在图像内，
    同时确保扩展后的矩形完全覆盖原始正方形。
    返回 (new_x, new_y, new_w, new_h)。
    """
    x, y, side, _ = square_bbox
    dw = int(side * expand_ratio)
    dh = dw  # 正方形，宽高扩展相同
    new_w = side + dw
    new_h = side + dh
    # 保持中心不变
    cx = x + side // 2
    cy = y + side // 2
    new_x = cx - new_w // 2
    new_y = cy - new_h // 2
    img_h, img_w = img_shape[:2]

    # 首先尝试整体平移，使扩展矩形尽量完全落在图像内
    # 平移范围：保证扩展矩形不超出图像，且仍包含原正方形
    # 左边界可移动范围：原正方形的左边界 (x) 必须 >= 新左边界，且新左边界 >= 0
    # 右边界：原正方形右边界 (x+side) <= 新右边界 (new_x+new_w) <= img_w
    # 因此 new_x 的取值范围是 [max(0, x+side-new_w), min(x, img_w-new_w)]
    new_x_min = max(0, x + side - new_w)   # 保证原正方形右边不超出新矩形右边
    new_x_max = min(x, img_w - new_w)      # 保证原正方形左边不超出新矩形左边，且新矩形不超出右边界
    if new_x_min <= new_x_max:
        new_x = (new_x_min + new_x_max) // 2
    else:
        # 如果没有合法范围（说明图像太小），则优先保证原正方形完整，允许裁切新矩形
        # 此时 new_x 取能使原正方形完全落在新矩形内的最接近值
        new_x = max(new_x_min, new_x_max)  # 实际上此时两者可能不等，取任意一个都能保证包含？我们保守一点
        new_x = min(max(new_x, 0), img_w - new_w) if new_w <= img_w else 0

    # 同样的逻辑处理 Y 方向
    new_y_min = max(0, y + side - new_h)
    new_y_max = min(y, img_h - new_h)
    if new_y_min <= new_y_max:
        new_y = (new_y_min + new_y_max) // 2
    else:
        new_y = min(max(new_y, 0), img_h - new_h) if new_h <= img_h else 0

    # 最终边界处理
    if new_w > img_w:
        new_w = img_w
        new_x = 0
    if new_h > img_h:
        new_h = img_h
        new_y = 0

    # 确保原正方形完全在新矩形内
    if new_x > x:
        new_x = x
    if new_y > y:
        new_y = y
    if new_x + new_w < x + side:
        new_w = (x + side) - new_x
    if new_y + new_h < y + side:
        new_h = (y + side) - new_y

    return new_x, new_y, new_w, new_h

def ensure_same_size(img, mask):
    """将 mask 缩放到与 img 相同的尺寸。"""
    h, w = img.shape[:2]
    mh, mw = mask.shape[:2]
    if (h, w) != (mh, mw):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

def resize_if_needed(img, max_long_edge=1280):
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def process_one(raw_path, mask_path, output_path, expand_ratio=0.2, max_long_edge=1280, auto_bright=True):
    # 1. 解码 RAW
    try:
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True,
                                  output_bps=8,
                                  no_auto_bright=not auto_bright,
                                  auto_bright_thr=0.03)
    except Exception as e:
        print(f"  RAW decode error: {e}")
        return False

    # 2. 读取并处理 mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  Mask not readable: {mask_path}")
        return False
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = ensure_same_size(rgb, mask)

    # 3. 获取人像外接矩形
    bbox = get_person_bbox(mask)
    if bbox is None:
        print(f"  No person found in mask, keep full image.")
        output_img = rgb
    else:
        # 4. 生成包含人像的正方形
        square = bbox_to_containing_square(bbox, rgb.shape)
        if square is None:
            print(f"  Warning: Cannot create square containing the person (image too small?), keep full image.")
            output_img = rgb
        else:
            ratio = square_area_ratio(square, rgb.shape)
            print(f"  Square area ratio: {ratio:.2%}")
            if ratio >= 0.6:
                output_img = rgb
                print(f"  No crop needed.")
            else:
                # 5. 扩展正方形并确保完整保留原正方形
                crop_bbox = expand_and_fit(square, rgb.shape, expand_ratio)
                x, y, w, h = crop_bbox
                output_img = rgb[y:y+h, x:x+w]
                print(f"  Cropped to expanded square ({w}x{h}).")

    # 6. 缩放
    output_img = resize_if_needed(output_img, max_long_edge)
    print(f"  Resized to {output_img.shape[1]}x{output_img.shape[0]}")

    # 7. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, output_img)
    print(f"  Saved: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--mask_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--mask_ext', default='.png')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--expand_ratio', type=float, default=0.3, help='Expansion ratio (default 0.3)')
    parser.add_argument('--max_long_edge', type=int, default=1440, help='Resize longest side to this value (default 1280)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_files = []
    for ext in RAW_EXTS:
        raw_files.extend(glob(os.path.join(args.raw_dir, '*' + ext)))
        raw_files.extend(glob(os.path.join(args.raw_dir, '*' + ext.upper())))
    raw_files = sorted(set(raw_files))

    if not raw_files:
        print(f"No RAW files found in {args.raw_dir}")
        return

    print(f"Found {len(raw_files)} RAW file(s)")

    success = 0
    for raw_path in raw_files:
        base = os.path.basename(raw_path)
        name = os.path.splitext(base)[0]
        mask_path = os.path.join(args.mask_dir, name + args.mask_ext)
        if not os.path.exists(mask_path):
            print(f"Mask not found for {base}, skip.")
            continue

        out_path = os.path.join(args.output_dir, name + '.png')
        if not args.overwrite and os.path.exists(out_path):
            print(f"Output exists: {out_path}, skip.")
            continue

        print(f"\nProcessing {base} ...")
        if process_one(raw_path, mask_path, out_path, args.expand_ratio, args.max_long_edge):
            success += 1

    print(f"\nDone. Successfully processed {success} of {len(raw_files)} files.")

if __name__ == '__main__':
    main()