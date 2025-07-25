import argparse
import os
import pandas as pd
import transformers

from benchmark.intrinsic.crows.crows_runner import CrowSPairsRunner

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertForMaskedLM",
    help="Model to evalute (e.g., BertForMaskedLM). Typically, these correspond to a HuggingFace "
    "class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    default="gender", # race-color / socioeconomic / sexual-orientation / religion / age / nationality / disabiltiy / physical-appearance
    help="Determines which CrowS-Pairs dataset split to evaluate against.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    print("Running CrowS-Pairs benchmark:")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")

    # Load model and tokenizer.
    model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = CrowSPairsRunner(
        model=model,
        tokenizer=tokenizer,
        input_file="benchmark/intrinsic/crows/crows_pairs_anonymized.csv",
        bias_type=args.bias_type,
        is_generative=False,  # Affects model scoring.
    )
    results = runner()

    print(f"Metric: {results}")
        # 추가된 코드: 결과를 CSV로 저장
    output_dir = os.path.join(args.persistent_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # output_file = os.path.join(output_dir, f"crows_results_{args.bias_type}.csv")
    # results_df = pd.DataFrame([results])
    # results_df.to_csv(output_file, index=False)

    # print(f"Results saved to {output_file}")
