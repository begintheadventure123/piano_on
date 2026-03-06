import argparse

from inference_utils import decision_threshold_from_pack, load_effective_config, load_model_pack, predict_probability


def main():
    parser = argparse.ArgumentParser(description="Run baseline model on one audio file")
    parser.add_argument("audio_path")
    parser.add_argument("--model", default="ml/models/baseline_logreg.joblib")
    parser.add_argument("--config", default="ml/configs/train_config.yaml")
    args = parser.parse_args()

    pack = load_model_pack(args.model)
    cfg = load_effective_config(pack, args.config)
    threshold = decision_threshold_from_pack(pack)
    p_raw, p = predict_probability(args.audio_path, pack, cfg)
    pred = int(p >= threshold)
    print(f"file={args.audio_path}")
    print(f"piano_probability_raw={p_raw:.4f}")
    print(f"piano_probability={p:.4f}")
    print(f"decision_threshold={threshold:.4f}")
    print(f"prediction={'piano' if pred else 'non_piano'}")


if __name__ == "__main__":
    main()
