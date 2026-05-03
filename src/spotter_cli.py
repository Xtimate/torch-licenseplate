import os
import subprocess
import sys
from calendar import c
from doctest import run_docstring_examples

from tomllib import load

sys.path.insert(0, os.path.dirname(__file__))

import yaml
from PIL import Image

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    print("Config saved")


def prompt(text, default=None):
    if default is not None:
        val = input(f"{text} (default: {default}): ").strip()
        return val if val else str(default)
    return input(f"{text}: ").strip()


def choose(options):
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        val = input("\n> ").strip()
        if val.isdigit() and 1 <= int(val) <= len(options):
            return int(val)
        print("Invalid choice, try again.")


def header():
    os.system("clear")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Spotter CLI")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


def run_train(config):
    header()
    print("[Train]\n")

    data_dir = config["training"]["data_dir"]
    index_path = os.path.join(data_dir, "index.txt")

    count = 0
    if os.path.exists(index_path):
        with open(index_path) as f:
            count = sum(1 for l in f if l.strip())

    if os.path.exists(index_path):
        with open(index_path) as f:
            count = sum(1 for l in f if l.strip())
            print(f"  ✓ Dataset found at {data_dir} ({count} samples)\n")
            print("  1. Use existing dataset and train")
            print("  2. Generate new dataset and train")
            choice = choose(
                ["Use existing dataset and train", "Generate new dataset and train"]
            )
            if choice == 2:
                data_dir = run_generate(config)
                if data_dir is None:
                    return
    else:
        print(f"No dataset found at {data_dir}\n")
        data_dir = run_generate(config)
        if data_dir is None:
            return

    print("\n[Training setup]\n")
    epochs = prompt("Epochs to train (recommended: 100)", default=100)

    print(f"\n Starting training on {data_dir} for {epochs} epochs\n")
    subprocess.run(
        [
            sys.executable,
            "src/train_recognizer.py",
            "--data-dir",
            data_dir,
            "--epochs",
            str(epochs),
            "--size",
            str(count),
        ]
    )


def run_generate(config):
    header()
    print("[Generate Dataset]\n")

    synthetic = prompt(
        "Synthetic plates to generate (recommended: 20000)", default=20000
    )
    real_dir = prompt("Real plates directory", default="eu-plates/eu-license-plates/eu")
    augment = prompt("Augmentations per real plate (recommended: 25)", default=25)
    out_dir = prompt("Output directory", default=config["training"]["data_dir"])

    print(f"\n  Generating dataset → {out_dir}...\n")
    result = subprocess.run(
        [
            sys.executable,
            "src/generate_mixed_dataset.py",
            "--real",
            real_dir,
            "--detector",
            config["models"]["detector"],
            "--synthetic",
            str(synthetic),
            "--augment",
            str(augment),
            "--out",
            out_dir,
        ]
    )

    if result.returncode != 0:
        print("Failed to generate dataset")
        return None

        config["training"]["data_dir"] = out_dir
        save_config(config)
        return out_dir


def run_recognize(config):
    header()
    print("[Recognize]\n")
    path = prompt("Path to image")
    if not os.path.exists(path):
        print("File not found")
        return

    import torch

    from dataset import CHARS
    from recognizer import (
        load_recognizer,
        load_recognizer_onnx,
        recognize_from_image,
        recognize_from_image_onnx,
    )

    img = Image.open(path).convert("RGB")
    runtime = config["runtime"]

    if runtime == "onnx":
        session = load_recognizer_onnx(config["models"]["recognizer"])
        result = recognize_from_image_onnx(
            img, session, threshold=config["inference"]["conf_threshold"]
        )
        print(f"\n  Text       : {result.text or '—'}")
        print(f"  Confidence : {result.confidence * 100:.0f}%")
        print(
            f"  Format     : {'✓ ' + (plate['country'] or '') if plate['valid_format'] else '✗'}"
        )
        if result.rejected:
            print(f"  Rejected   : {result.rejection_reason}")
    else:
        device = torch.device(str(config["inference"]["device"]))  # type: ignore
        model = load_recognizer(len(CHARS), config["models"]["recognizer_pt"], device)
        text = recognize_from_image(img, model, device)
        print(f"\n  Text       : {text or '—'}")


def run_detect(config):
    header()
    print("[Detect]\n")
    path = prompt("Path to image")
    if not os.path.exists(path):
        print("File not found")
        return

    from detector import detect_from_image, load_detector, load_detector_onnx

    img = Image.open(path).convert("RGB")

    if config["runtime"] == "onnx":
        model = load_detector_onnx(config["models"]["detector"])
    else:
        model = load_detector(config["models"]["detector"])

    detections = detect_from_image(
        model, img, conf=config["inference"]["conf_threshold"]
    )
    print(f"\n Found {len(detections)} plate(s):\n")
    for d in detections:
        print(f"[{d['x1']},{d['y1']},{d['x2']},{d['y2']}] conf: {d['conf']:.2f}")


def run_pipeline(config, batch=False):
    header()
    print(f"[ {'Batch ' if batch else ''}Pipeline ]\n")

    if batch:
        paths = prompt("Image paths (comma-separated)").split(",")
        paths = [p.strip() for p in paths]
    else:
        paths = [prompt("Path to image")]

    from detector import load_detector, load_detector_onnx
    from pipeline import run_pipeline as _run
    from pipeline import run_pipeline_batch

    if config["runtime"] == "onnx":
        detector = load_detector_onnx(config["models"]["detector"])
        from recognizer import load_recognizer_onnx

        recognizer = load_recognizer_onnx(config["models"]["recognizer"])
    else:
        import torch

        from dataset import CHARS

        detector = load_detector(config["models"]["detector_pt"])
        from recognizer import load_recognizer

        device = torch.device(config["inference"]["device"])  # type: ignore
        recognizer = load_recognizer(
            len(CHARS), config["models"]["recognizer_pt"], device
        )

    images = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: {p} does not exist, skipping")
            continue
        images.append(Image.open(p).convert("RGB"))

    if not images:
        return
    if batch:
        results = run_pipeline_batch(detector, recognizer, images)
        for r in results:
            print(f"Image {r['image_index'] + 1}:")
            for plate in r["plates"]:
                print(
                    f"    {plate['text']}  conf: {plate.get('confidence', 0) * 100:.0f}%  format: {'✓ ' + plate['country'] if plate['valid_format'] else '✗'}"
                )
    else:
        results = _run(detector, recognizer, images[0], None)
        print()
        for plate in results:
            print(
                f"  {plate['text']}  conf: {plate.get('confidence', 0) * 100:.0f}%  format: {'✓ ' + plate['country'] if plate['valid_format'] else '✗'}"
            )


def run_configure(config):
    header()
    print("[ Configure ]\n")
    print("  Current settings:\n")
    print(f"  1. Runtime          : {config['runtime']}")
    print(f"  2. Device           : {config['inference']['device']}")
    print(f"  3. Conf threshold   : {config['inference']['conf_threshold']}")
    print(f"  4. Detector (ONNX)  : {config['models']['detector']}")
    print(f"  5. Recognizer (ONNX): {config['models']['recognizer']}")
    print(f"  6. Detector (PT)    : {config['models']['detector_pt']}")
    print(f"  7. Recognizer (PT)  : {config['models']['recognizer_pt']}")
    print(f"  8. Training data    : {config['training']['data_dir']}")
    print(f"  9. Batch size       : {config['training']['batch_size']}")
    print(f"  10. Learning rate   : {config['training']['lr']}")
    print(f"  0. Back\n")

    val = input("> ").strip()
    mapping = {
        "1": ("runtime", None),
        "2": ("inference.device", None),
        "3": ("inference.conf_threshold", float),
        "4": ("models.detector", None),
        "5": ("models.recognizer", None),
        "6": ("models.detector_pt", None),
        "7": ("models.recognizer_pt", None),
        "8": ("training.data_dir", None),
        "9": ("training.batch_size", int),
        "10": ("training.lr", float),
    }

    if val == "0" or val not in mapping:
        return

    key, cast = mapping[val]
    new_val = input(f"  New value: ").strip()
    if cast:
        new_val = cast(new_val)

    parts = key.split(".")
    if len(parts) == 2:
        config[parts[0]][parts[1]] = new_val
    else:
        config[parts[0]] = new_val

    save_config(config)


def main():
    config = load_config()

    while True:
        header()
        print(
            f"  Runtime: {config['runtime']}  |  Device: {config['inference']['device']}\n"
        )
        print("  1. Train")
        print("  2. Recognize")
        print("  3. Detect")
        print("  4. Pipeline")
        print("  5. Batch pipeline")
        print("  6. Configure")
        print("  7. Exit\n")

        val = input("\n> ").strip()
        while not val.isdigit() or not (1 <= int(val) <= 7):
            val = input("> ").strip()
        choice = int(val)

        if choice == 1:
            run_train(config)
        elif choice == 2:
            run_recognize(config)
        elif choice == 3:
            run_detect(config)
        elif choice == 4:
            run_pipeline(config, batch=False)
        elif choice == 5:
            run_pipeline(config, batch=True)
        elif choice == 6:
            run_configure(config)
            config = load_config()
        elif choice == 7:
            print("\n  Bye.\n")
            break

        input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()
