"""Simple CLI for `process_csv.py` actions.

Usage examples:
  python cli.py summary c:/path/to/Telco_Cusomer_Churn.csv
  python cli.py clean c:/path/to/Telco_Cusomer_Churn.csv --out cleaned.csv
  python cli.py features c:/path/to/cleaned.csv
"""
import argparse
import json
import sys

from process_csv import load_csv, summarize, clean, prepare_features, save_cleaned


def cmd_summary(args):
    df = load_csv(args.input)
    s = summarize(df)
    print(json.dumps(s, indent=2, default=str))


def cmd_clean(args):
    df = load_csv(args.input)
    dfc = clean(df)
    save_cleaned(dfc, args.out)
    print("Cleaned file written to", args.out)


def cmd_features(args):
    df = load_csv(args.input)
    dfc = clean(df)
    X, y = prepare_features(dfc)
    print(f"Features shape: {X.shape}; Target length: {len(y)}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("summary")
    p.add_argument("input")

    p = sub.add_parser("clean")
    p.add_argument("input")
    p.add_argument("--out", default="cleaned_telco.csv")

    p = sub.add_parser("features")
    p.add_argument("input")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    if args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "clean":
        cmd_clean(args)
    elif args.cmd == "features":
        cmd_features(args)


if __name__ == "__main__":
    main()
