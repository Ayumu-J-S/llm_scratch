from __future__ import annotations

import argparse
import json

from runtime.environment import collect_environment, unmet_requirements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report the llm_scratch execution environment")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--require-cuda", action="store_true", help="fail unless CUDA is available")
    parser.add_argument(
        "--require-bf16", action="store_true", help="fail unless CUDA BF16 is supported"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_environment()
    failures = unmet_requirements(
        report,
        require_cuda=args.require_cuda,
        require_bf16=args.require_bf16,
    )
    report["requirements"] = {
        "require_cuda": args.require_cuda,
        "require_bf16": args.require_bf16,
        "passed": not failures,
        "failures": failures,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        cuda = report["cuda"]
        print(f"host: {report['host']} ({report['architecture']})")
        print(f"OS: {report['os']}")
        print(f"Python: {report['python']}")
        print(
            f"PyTorch: {report['torch']['version']} "
            f"(CUDA build {report['torch']['compiled_cuda']}, {report['torch']['module']})"
        )
        print(
            f"CUDA: available={cuda['available']} runtime={cuda['runtime_version']} "
            f"driver={cuda['driver_version']} BF16={cuda['bf16_supported']}"
        )
        for device in cuda["devices"]:
            print(f"GPU {device['index']}: {device['name']} compute={device['compute_capability']}")
        print(report["memory_interpretation"])
        if failures:
            print("requirements failed: " + "; ".join(failures))

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
