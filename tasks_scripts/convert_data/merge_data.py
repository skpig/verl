import pandas as pd
import sys
from verl.utils.fs import copy_local_path_from_hdfs


def concatenate_files(input_files, output_file):
    data_frames = []
    for file in input_files:
        try:
            local_path = copy_local_path_from_hdfs(file)
            df = pd.read_parquet(local_path)
            data_frames.append(df)
        except FileNotFoundError:
            print(f"File {file} not found")
        except Exception as e:
            print(f"Error when reading {file}: {e}")

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        try:
            combined_df.to_parquet(output_file)
            print(f"Merged file saved to {output_file}.")
        except Exception as e:
            print(f"Error when writing {output_file}")
    else:
        print("No input files, do nothing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="merge parquet files, usage: python3 merge_data.py a.parquet b.parquet --output_file merged.parquet"
    )
    parser.add_argument("input_files", nargs="+", help="input parquet files to merge")
    parser.add_argument("--output_file", help="output parquet file path", required=True)

    args = parser.parse_args()

    input_files = args.input_files
    output_file = args.output_file

    concatenate_files(input_files, output_file)
