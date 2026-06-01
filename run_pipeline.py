import argparse
import os
from src.train import run_training
from src.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="MrScraper Price Intelligence Multi-Approach Pipeline CLI")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"], 
                        help="Execution mode: 'train' to fit models, 'infer' to predict on test data.")
    parser.add_argument("--input", type=str, help="Path to input test CSV (Required if mode is 'infer').")
    parser.add_argument("--output", type=str, default="submission_output.csv", 
                        help="Destination path for predictions CSV.")
    parser.add_argument("--approach", type=str, default="global", choices=["global", "entity"],
                        help="Modeling strategy to use for inference: 'global' (Approach 1) or 'entity' (Approach 2).")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training()
    elif args.mode == "infer":
        if not args.input:
            parser.error("--input is required when --mode is set to 'infer'")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"[ERROR] Input test file not found at: {args.input}")
        run_inference(args.input, args.output, args.approach)

if __name__ == "__main__":
    main()