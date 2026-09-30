from pathlib import Path
from ultralytics import YOLO


def main():

    # ---------------------------------------------------------
    # PROJECT PATH
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent

    data_yaml = project_root / "data.yaml"
    model_path = project_root / "yolo11n.pt"
    runs_dir = project_root / "runs"


    # ---------------------------------------------------------
    # INFORMATION
    # ---------------------------------------------------------

    print("=" * 60)
    print("LICENSE PLATE DETECTOR TRAINING")
    print("=" * 60)

    print(f"Dataset : {data_yaml}")
    print(f"Model   : {model_path}")
    print("Device  : NVIDIA GeForce RTX 3050")
    print("=" * 60)


    # ---------------------------------------------------------
    # LOAD PRETRAINED YOLO11
    # ---------------------------------------------------------

    model = YOLO(str(model_path))


    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("\nStarting GPU training...\n")

    results = model.train(

        # Dataset
        data=str(data_yaml),

        # Training
        epochs=3,
        imgsz=640,

        # RTX 3050 4GB
        batch=8,

        # Windows
        workers=2,

        # NVIDIA GPU
        device=0,

        # Output
        project=str(runs_dir),
        name="license_plate_detector",

        # Save model
        save=True,

        # Early stopping
        patience=20,

        # Augmentation
        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        # Disable mosaic near end
        close_mosaic=10,

        # Don't cache dataset
        cache=False,

        # Show training information
        verbose=True
    )


    # ---------------------------------------------------------
    # COMPLETED
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("\nYour trained model should be here:")

    print(
        runs_dir /
        "license_plate_detector" /
        "weights" /
        "best.pt"
    )


if __name__ == "__main__":
    main()