"""
Runs a training loop with specified inputs.
"""

# %% Imports

import argparse
import pandas as pd

from training.train import train, save_artifacts


# %%

def main():

    # Craft and parse arguments

    parser = argparse.ArgumentParser(
        description='Train and save a policy_area classifier'
    )
    parser.add_argument(
        '--input', '-i', required=False,
        type=str,
        default='input_data.parquet',
        help='Path to .parquet data'
    )
    parser.add_argument(
        '--save_folder', '-s', required=False,
        default='model_artifacts',
        help='Folder to save .joblib files'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Fraction of data to hold out for testing'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()


    # Read the specified input data file

    df = pd.read_parquet(args.input)
    texts = df['bill_text'].fillna('').values
    labels = df['policy_area'].fillna('').values


    # Execute the training function and save results

    model, vect, le = train(
        texts, labels,
        test_size=args.test_size,
        random_state=args.random_state
    )
    save_artifacts(
        args.save_folder,
        model, vect, le
    )

if __name__ == '__main__':
    main()