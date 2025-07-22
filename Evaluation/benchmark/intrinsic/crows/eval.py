import argparse
import os
import torch
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
    help="Model to evaluate (e.g., BertForMaskedLM). Typically, these correspond to a HuggingFace class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    default="gender",
    help="Determines which CrowS-Pairs dataset split to evaluate against.",
)
parser.add_argument(
    "--layer_indices",
    nargs='+',
    type=int,
    default=[11],  # Default to layers 10, 11, 12 (zero-based indexing)
    help="Indices of the layers to use (zero-based). For example, to use layers 10-12, use --layer_indices 9 10 11.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    print("Running CrowS-Pairs benchmark:")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - layers used: {args.layer_indices}")

    # Load the full model and tokenizer
    full_model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    full_model.eval()

    # Create a new configuration with num_hidden_layers set to the number of layers you're using
    config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_hidden_layers = len(args.layer_indices)

    # Create a new model with this configuration
    model = transformers.BertForMaskedLM(config)

    # Copy the embeddings from the full model
    model.bert.embeddings = full_model.bert.embeddings

    # Copy the specified layers
    selected_layers = args.layer_indices  # e.g., [9, 10, 11] for layers 10-12
    model.bert.encoder.layer = torch.nn.ModuleList(
        [full_model.bert.encoder.layer[i] for i in selected_layers]
    )

    # Copy the prediction head
    model.cls = full_model.cls

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    runner = CrowSPairsRunner(
        model=model,
        tokenizer=tokenizer,
        input_file="benchmark/intrinsic/crows/crows_pairs_anonymized.csv",
        bias_type=args.bias_type,
        is_generative=False,  # Affects model scoring.
    )
    results = runner()

    print(f"Metric: {results}")
