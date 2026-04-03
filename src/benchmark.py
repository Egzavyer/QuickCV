from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def find_image_ids(image_dir: Path) -> list[str]:
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [p.stem for p in files]


def build_command(
    python_exe: str,
    main_script: Path,
    fast_model: str,
    slow_model: str,
    image_dir: Path,
    image_ids: list[str],
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_exe,
        str(main_script),
        '--fast-model', fast_model,
        '--slow-model', slow_model,
        '--image-dir', str(image_dir),
        '--threshold', str(args.threshold),
        '--min-box-area', str(args.min_box_area),
        '--stage1-imgsz', str(args.stage1_imgsz),
        '--stage2-imgsz', str(args.stage2_imgsz),
        '--min-pad', str(args.min_pad),
        '--pad-ratio', str(args.pad_ratio),
        '--similar-min-conf', str(args.similar_min_conf),
        '--similar-min-delta', str(args.similar_min_delta),
        '--general-min-conf', str(args.general_min_conf),
        '--general-min-delta', str(args.general_min_delta),
        '--images',
        *image_ids,
    ]
    if args.save_vis:
        cmd.append('--save-vis')
    return cmd


def run_command(cmd: list[str], log_path: Path) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stdout
    if proc.stderr:
        text += ('\n' if text and not text.endswith('\n') else '') + proc.stderr
    log_path.write_text(text, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(
            f'Command failed with exit code {proc.returncode}. See {log_path}'
        )
    return text


def parse_summary(text: str) -> dict[str, Any]:
    patterns = {
        'images_processed': r'Images processed: (\d+)',
        'stage1_detections': r'Stage-1 detections kept: (\d+)',
        'escalated': r'Escalated detections: (\d+) \(([-+]?\d+(?:\.\d+)?)%\)',
        'improved': r'Stage-2 improved confidence: (\d+) \(([-+]?\d+(?:\.\d+)?)% of escalations\)',
        'stage2_no_detection': r'Stage-2 returned no detection: (\d+)',
        'stage2_agree': r'Stage-2 agreed with stage 1: (\d+)',
        'accepted_relabels': r'Accepted relabels: (\d+)',
        'rejected_relabels': r'Rejected relabels: (\d+)',
        'rejected_similar': r'Rejected similar-class relabels: (\d+)',
        'stage1_ms': r'Average stage-1 inference time: ([-+]?\d+(?:\.\d+)?) ms/image',
        'stage2_ms': r'Average stage-2 inference time: ([-+]?\d+(?:\.\d+)?) ms/crop',
        'stage1_conf_all': r'Average stage-1 confidence \(all detections\): ([-+]?\d+(?:\.\d+)?)',
        'stage1_conf_escalated': r'Average stage-1 confidence \(escalated only\): ([-+]?\d+(?:\.\d+)?)',
        'stage2_conf_best': r'Average stage-2 best confidence: ([-+]?\d+(?:\.\d+)?)',
        'delta': r'Average confidence delta \(stage2 - stage1\): ([-+]?\d+(?:\.\d+)?)',
        'total_wall_ms': r'Total wall-clock time: ([-+]?\d+(?:\.\d+)?) ms',
    }
    out: dict[str, Any] = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            out[key] = None
        elif len(m.groups()) == 1:
            val = m.group(1)
            out[key] = float(val) if '.' in val else int(val)
        else:
            out[key] = tuple(
                float(g) if '.' in g else int(g) for g in m.groups()
            )
    return out


def markdown_table(results: dict[str, dict[str, Any]]) -> str:
    headers = [
        'Config', 'Imgs', 'Escalations', 'Stage1 ms/img', 'Stage2 ms/crop',
        'Wall ms', 'Avg Δconf', 'No-detect'
    ]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for name, r in results.items():
        esc = r.get('escalated')
        if isinstance(esc, tuple):
            esc_text = f'{esc[0]} ({esc[1]:.2f}%)'
        else:
            esc_text = 'n/a'
        lines.append(
            '| ' + ' | '.join([
                name,
                str(r.get('images_processed', 'n/a')),
                esc_text,
                fmt(r.get('stage1_ms')),
                fmt(r.get('stage2_ms')),
                fmt(r.get('total_wall_ms')),
                fmt(r.get('delta')),
                str(r.get('stage2_no_detection', 'n/a')),
            ]) + ' |'
        )
    return '\n'.join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.2f}'
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run warm-up and larger hybrid benchmark runs.')
    parser.add_argument('--main-script', default='main.py')
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--fast-model', required=True)
    parser.add_argument('--slow-model', required=True)
    parser.add_argument('--fast-model-int8')
    parser.add_argument('--slow-model-int8')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--min-box-area', type=float, default=0.0)
    parser.add_argument('--stage1-imgsz', type=int, default=640)
    parser.add_argument('--stage2-imgsz', type=int, default=320)
    parser.add_argument('--min-pad', type=int, default=32)
    parser.add_argument('--pad-ratio', type=float, default=0.25)
    parser.add_argument('--similar-min-conf', type=float, default=0.85)
    parser.add_argument('--similar-min-delta', type=float, default=0.20)
    parser.add_argument('--general-min-conf', type=float, default=0.75)
    parser.add_argument('--general-min-delta', type=float, default=0.15)
    parser.add_argument('--out-dir', default='benchmark_runs')
    parser.add_argument('--save-vis', action='store_true')
    args = parser.parse_args()

    main_script = Path(args.main_script)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids = find_image_ids(image_dir)
    if len(image_ids) < 2:
        raise SystemExit('Need at least 2 images for warm-up + benchmark.')

    warmup_id = image_ids[0]
    benchmark_ids = image_ids[1: args.count + 1]
    if not benchmark_ids:
        raise SystemExit('No images left for benchmark after warm-up.')

    configs: list[tuple[str, str, str]] = [('regular_openvino', args.fast_model, args.slow_model)]
    if args.fast_model_int8 and args.slow_model_int8:
        configs.append(('int8_openvino', args.fast_model_int8, args.slow_model_int8))

    all_results: dict[str, dict[str, Any]] = {}

    for name, fast_model, slow_model in configs:
        print(f'\n=== {name}: warm-up on {warmup_id} ===')
        warm_cmd = build_command(
            sys.executable,
            main_script,
            fast_model,
            slow_model,
            image_dir,
            [warmup_id],
            args,
        )
        warm_log = out_dir / f'{name}_warmup.log'
        run_command(warm_cmd, warm_log)
        print(f'Warm-up log saved to {warm_log}')

        print(f'=== {name}: benchmark on {len(benchmark_ids)} images ===')
        bench_cmd = build_command(
            sys.executable,
            main_script,
            fast_model,
            slow_model,
            image_dir,
            benchmark_ids,
            args,
        )
        bench_log = out_dir / f'{name}_benchmark.log'
        text = run_command(bench_cmd, bench_log)
        result = parse_summary(text)
        all_results[name] = result
        print(f'Benchmark log saved to {bench_log}')
        print(markdown_table({name: result}))

    table = markdown_table(all_results)
    table_path = out_dir / 'summary_table.md'
    table_path.write_text(table + '\n', encoding='utf-8')

    print('\n=== Combined summary ===')
    print(table)
    print(f'\nSaved summary table to {table_path}')


if __name__ == '__main__':
    main()
