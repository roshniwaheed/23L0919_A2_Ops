
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

AUDIT_PATH = Path("reports/dataset_audit.csv")
OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(AUDIT_PATH)

    # Remove corrupt files and exact duplicate copies
    df = df[
        (~df["corrupt"]) &
        (~df["duplicate"])
    ].copy()

    # Retain only required columns
    df = df[["path", "class", "label", "sha256"]]

    # Check file existence
    missing = [
        path for path in df["path"]
        if not Path(path).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image paths are missing. "
            f"Example: {missing[0]}"
        )

    # Ensure unique image contents
    assert df["sha256"].is_unique

    # First split: 70% training, 30% remaining
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"] #ensures that the class proportions remain similar in all three subsets.
    )

    # Second split: 15% validation, 15% testing
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_df["label"]
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    for name, split_df in splits.items():
        split_df = split_df.sort_values(
            "path"
        ).reset_index(drop=True)

        split_df.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False
        )

        print(f"\n{name.upper()} SET")
        print("Total:", len(split_df))
        print(split_df["class"].value_counts())

    # Verify no content leakage between splits
    train_hashes = set(train_df["sha256"])
    val_hashes = set(val_df["sha256"])
    test_hashes = set(test_df["sha256"])

    assert train_hashes.isdisjoint(val_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert val_hashes.isdisjoint(test_hashes)

    assert (
        len(train_df) + len(val_df) + len(test_df)
        == len(df)
    )

    print("\nNo exact duplicate leakage detected.")
    print("All splits generated successfully.")


if __name__ == "__main__":
    main()
