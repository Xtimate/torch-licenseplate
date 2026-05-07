import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def export_labeled_crops(out_dir: str) -> int:
    from api.database import get_labeled_items

    os.makedirs(out_dir, exist_ok=True)

    items = get_labeled_items()
    if not items:
        print("No labeled items in the review queue yet.")
        return 0

    index_path = os.path.join(out_dir, "index.txt")

    existing = set()
    if os.path.exists(index_path):
        with open(index_path) as f:
            for line in f:
                existing.add(line.strip().split(" ")[0])

    new_count = 0
    with open(index_path, "a") as index_file:
        for item in items:
            filename = f"queue_{item['id']}_{item['labeled_text']}.jpg"

            if filename in existing:
                continue

            filepath = os.path.join(out_dir, filename)
            with open(filepath, "wb") as f:
                f.write(item["crop"])

            clean_text = item["labeled_text"].replace("", "_")
            index_file.write(f"{filename} {clean_text}\n")
            new_count += 1
            print(f"Exported {filename} -> {clean_text}")

    print(f"Exported {new_count} new crops to {out_dir}.")
    return new_count


def merge_index_files(base_dir: str, queue_dir: str, merged_dir: str):
    os.makedirs(merged_dir, exist_ok=True)
    merged_index_path = os.path.join(merged_dir, "index.txt")

    entries = []

    base_index = os.path.join(base_dir, "index.txt")
    if os.path.exists(base_index):
        with open(base_index) as f:
            for line in f:
                line = line.strip()
                if line:
                    filename, text = line.split(" ", 1)
                    abs_path = os.path.abspath(os.path.join(base_dir, filename))
                    entries.append((abs_path, text))

    queue_index = os.path.join(queue_dir, "index.txt")
    queue_count = 0
    if os.path.exists(queue_index):
        with open(queue_index) as f:
            for line in f:
                line = line.strip()
                if line:
                    filename, text = line.split(" ", 1)
                    abs_path = os.path.abspath(os.path.join(queue_dir, filename))
                    entries.append((abs_path, text))
                    queue_count += 1

    print(f"Queue crops: {queue_count} entries from {queue_dir}.")

    with open(merged_index_path, "w") as f:
        for path, text in entries:
            f.write(f"{path} {text}\n")

    print(
        f"Merged index: {len(entries)} total entries entries saved to {merged_index_path}."
    )

    return len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/mixed")
    parser.add_argument("--queue-dir", type=str, default="data/queue_exports")
    parser.add_argument("--merged-dir", type=str, default="data/merged")
    parser.add_argument("--epochs", type=str, default=20)
    parser.add_argument("--skip-retrain", action="store_true")
    args = parser.parse_args()

    print("\n── Step 1: Export labeled crops from review queue ──")
    exported = export_labeled_crops(args.queue_dir)
    if exported == 0 and not os.path.exists(os.path.join(args.queue_dir, "index.txt")):
        print("Nothing to retrain on. Label some plates in the review tab first.")
        sys.exit(0)

    print("\n── Step 2: Merge with base dataset ──")
    total = merge_index_files(args.data_dir, args.queue_dir, args.merged_dir)

    if args.skip_retrain:
        print("\nSkipping retrain (--skip-retrain flag set).")
        sys.exit(0)

    print(f"\n── Step 3: Retrain on {total} samples for {args.epochs} epochs ──")
    # Call train_recognizer.py as a subprocess so it handles its own device setup
    retrain_cmd = [
        sys.executable,
        "src/train_recognizer.py",
        "--data-dir",
        args.merged_dir,
        "--epochs",
        str(args.epochs),
        "--size",
        str(total),
    ]
    print(f"  Running: {' '.join(retrain_cmd)}\n")
    subprocess.run(retrain_cmd, check=True)

    print("\n── Step 4: Export to ONNX ──")
    export_cmd = [
        sys.executable,
        "scripts/export_onnx.py",
    ]
    print(f"  Running: {' '.join(export_cmd)}\n")
    subprocess.run(export_cmd, check=True)

    print("\n── Step 5: Reload model in running service ──")
    # Signal the systemd service to restart and pick up the new ONNX model
    # Only runs if the service exists (i.e. on the droplet, not locally)
    result = subprocess.run(
        ["systemctl", "is-active", "spotter"], capture_output=True, text=True
    )
    if result.stdout.strip() == "active":
        subprocess.run(["systemctl", "restart", "spotter"], check=True)
        print("  Service restarted — new model is live.")
    else:
        print("  Spotter service not running — skipping restart.")
        print("  Manually run: systemctl restart spotter")

        print("\n── Done ──")
        print("  New model is live at onnx/lprnet.onnx")
        print("\n── Step 4b: Register model version ──")
        # Read the best loss from the training run
        # train_recognizer.py saves it to checkpoints/best_loss.txt
        loss = None
        loss_path = "checkpoints/best_loss.txt"
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss = float(f.read().strip())

        from api.database import get_labeled_items, register_model_version

        labeled_count = len(get_labeled_items())

        # Version filename based on timestamp
        from datetime import datetime

        version_name = f"lprnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"

        # Copy the exported ONNX to a versioned backup
        import shutil

        os.makedirs("onnx/versions", exist_ok=True)
        shutil.copy("onnx/lprnet.onnx", f"onnx/versions/{version_name}")

        register_model_version(
            filename=version_name,
            loss=loss or 0.0,
            labeled_samples=labeled_count,
        )
        print(
            f"  Registered version: {version_name} (loss={loss}, samples={labeled_count})"
        )
