import argparse
import math
import os
import random
from pathlib import Path

def write_pgm(path, width, height, pixels):
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode())
        f.write(bytes(pixels))

def make_image(width, height, seed):
    random.seed(seed)
    cx = width * (0.25 + 0.5 * random.random())
    cy = height * (0.25 + 0.5 * random.random())
    radius = min(width, height) * (0.12 + 0.12 * random.random())
    angle = random.random() * math.pi

    pixels = []
    for y in range(height):
        for x in range(width):
            # Smooth background gradient
            value = int(40 + 60 * x / max(1, width - 1))

            # Bright circle
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy < radius * radius:
                value += 130

            # Diagonal stripe
            u = x * math.cos(angle) + y * math.sin(angle)
            if int(u / 10) % 2 == 0:
                value += 35

            # Small deterministic noise
            value += random.randint(-8, 8)
            pixels.append(max(0, min(255, value)))

    return pixels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", default="data/input")
    args = parser.parse_args()

    if args.count <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("count, width, and height must be positive")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        pixels = make_image(args.width, args.height, i)
        write_pgm(out / f"image_{i:04d}.pgm",
                   args.width, args.height, pixels)

    print(f"Generated {args.count} images of size "
          f"{args.width}x{args.height} in {out}")

if __name__ == "__main__":
    main()
