import os
import pandas as pd


def parse_results(hypersim_dir):
    result_csvs = sorted([x for x in os.listdir(hypersim_dir) if "_results.csv" in x])

    results = []
    for result_csv in result_csvs:
        result = pd.read_csv(os.path.join(hypersim_dir, result_csv))
        results.append(result)

    # Concatenate all results
    results = pd.concat(results, ignore_index=True)

    return results


def main(hypersim_dir):
    results = parse_results(hypersim_dir)

    # for each column, print the mean value
    for column in results.columns:
        print(f"{column}: {results[column].mean()}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
