"""CLI entry point: python -m mle_star [--resume] [--max-iter N] [--force-data]"""

import argparse
import os
import sys
import time

from . import config


def main() -> int:
    ap = argparse.ArgumentParser(description="MLE-STAR autonomous ML agent")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint/experiment store (C3)")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="override MAX_ITERATIONS for this run")
    ap.add_argument("--force-data", action="store_true",
                    help="rebuild the prepared-data cache")
    args = ap.parse_args()

    if not (os.path.exists(config.TRAIN_CSV) and os.path.exists(config.STORE_CSV)):
        print(f"Missing {config.TRAIN_CSV} / {config.STORE_CSV} in the working directory.")
        return 1

    if args.max_iter is not None:
        config.MAX_ITERATIONS = args.max_iter

    from . import agents, data, graph, llm

    config.ensure_dirs()
    data.prepare(force=args.force_data)

    overrides = agents.init_run(resume=args.resume)
    state = {
        "task": "Rossmann Store Sales forecasting (train.csv + store.csv)",
        **overrides,
    }

    print(f"Starting MLE-STAR v2 (max_iter={config.MAX_ITERATIONS}, "
          f"metric={config.METRIC}, beam={config.BEAM_WIDTH})")
    t0 = time.time()
    # recursion_limit: each iteration is one graph step plus overhead.
    result = graph.build().invoke(
        state, config={"recursion_limit": config.MAX_ITERATIONS * 3 + 20}
    )

    final = result.get("final", {})
    print("\n" + "=" * 60)
    print(f"Done in {(time.time() - t0) / 60:.1f} min | stop: {result.get('stop_reason')}")
    if "chosen_node" in final:
        chosen_test = final.get("test_scores", {}).get(str(final["chosen_node"]), {})
        print(f"Best single : val {config.METRIC} = {final.get('chosen_val'):.4f} | "
              f"test {config.METRIC} = {chosen_test.get(config.METRIC, float('nan')):.4f}")
        ens = final.get("ensemble") or {}
        if ens:
            print(f"Ensemble    : test {config.METRIC} = "
                  f"{ens['test_scores'][config.METRIC]:.4f} (members {ens['members']})")
    print(f"Tokens      : {llm.LEDGER.summary()}")
    print(f"Artifacts   : {config.OUTPUT_DIR}/ (model, report, results, experiments.jsonl)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
