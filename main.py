from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import TypedDict

import cv2  # type: ignore

from models.Model import Model


SIMILAR_CLASS_PAIRS = {
    ("Cyclist", "Pedestrian"),
    ("Pedestrian", "Cyclist"),
}


class Metrics(TypedDict):
    images_processed: int
    stage1_detections: int
    escalated: int
    improved: int
    stage2_no_detection: int
    stage2_agree: int
    accepted_relabels: int
    rejected_relabels: int
    kept_stage1_similar: int
    stage1_ms: list[float]
    stage2_ms: list[float]
    stage1_conf_all: list[float]
    stage1_conf_escalated: list[float]
    stage2_conf_best: list[float]
    confidence_delta: list[float]


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def print_summary(metrics: Metrics, total_wall_ms: float) -> None:
    stage1_detections = int(metrics["stage1_detections"])
    escalated = int(metrics["escalated"])
    improved = int(metrics["improved"])
    stage2_no_detection = int(metrics["stage2_no_detection"])

    escalation_rate = (
        (100.0 * escalated / stage1_detections) if stage1_detections else 0.0
    )
    improvement_rate = (100.0 * improved / escalated) if escalated else 0.0

    print("\n" + "=" * 72)
    print("PIPELINE SUMMARY")
    print("=" * 72)
    print(f"Images processed: {metrics['images_processed']}")
    print(f"Stage-1 detections kept: {stage1_detections}")
    print(f"Escalated detections: {escalated} ({escalation_rate:.2f}%)")
    print(
        f"Stage-2 improved confidence: {improved} ({improvement_rate:.2f}% of escalations)"
    )
    print(f"Stage-2 returned no detection: {stage2_no_detection}")
    print(f"Stage-2 agreed with stage 1: {metrics['stage2_agree']}")
    print(f"Accepted relabels: {metrics['accepted_relabels']}")
    print(f"Rejected relabels: {metrics['rejected_relabels']}")
    print(f"Rejected similar-class relabels: {metrics['kept_stage1_similar']}")
    print(
        f"Average stage-1 inference time: {safe_mean(metrics['stage1_ms']):.2f} ms/image"
    )
    print(
        f"Average stage-2 inference time: {safe_mean(metrics['stage2_ms']):.2f} ms/crop"
    )
    print(
        f"Average stage-1 confidence (all detections): {safe_mean(metrics['stage1_conf_all']):.4f}"
    )
    print(
        f"Average stage-1 confidence (escalated only): {safe_mean(metrics['stage1_conf_escalated']):.4f}"
    )
    print(
        f"Average stage-2 best confidence: {safe_mean(metrics['stage2_conf_best']):.4f}"
    )
    print(
        f"Average confidence delta (stage2 - stage1): {safe_mean(metrics['confidence_delta']):.4f}"
    )
    print(f"Total wall-clock time: {total_wall_ms:.2f} ms")
    print("=" * 72)


def resolve_images(image_dir: str, image_ids: list[str]) -> list[Path]:
    directory = Path(image_dir)
    return [directory / f"{image_id}.png" for image_id in image_ids]


def choose_final_detection(
    stage1_class: str,
    stage1_conf: float,
    best: dict | None,
    similar_min_conf: float,
    similar_min_delta: float,
    general_min_conf: float,
    general_min_delta: float,
) -> tuple[str, float, str, str | None]:
    if best is None:
        return stage1_class, stage1_conf, "stage2_no_detection", None

    stage2_class = str(best["class_name"])
    stage2_conf = float(best["confidence"])
    delta = stage2_conf - stage1_conf

    if stage2_class == stage1_class:
        return stage2_class, stage2_conf, "agree", stage2_class

    if (stage1_class, stage2_class) in SIMILAR_CLASS_PAIRS:
        if stage2_conf >= similar_min_conf and delta >= similar_min_delta:
            return stage2_class, stage2_conf, "similar_class_relabel", stage2_class
        return stage1_class, stage1_conf, "kept_stage1_similar", stage2_class

    if stage2_conf >= general_min_conf and delta >= general_min_delta:
        return stage2_class, stage2_conf, "relabel", stage2_class

    return stage1_class, stage1_conf, "kept_stage1", stage2_class


def draw_stage2_on_image(image, item: dict, best: dict | None, final: dict):
    x1, y1, x2, y2 = [int(v) for v in item["crop_xyxy"]]
    reason = str(final["reason"])

    color = (0, 255, 255)
    if reason in {"relabel", "similar_class_relabel"}:
        color = (0, 165, 255)
    elif reason in {"kept_stage1", "kept_stage1_similar", "stage2_no_detection"}:
        color = (255, 255, 0)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    if best is None:
        label = (
            f"ESC {reason} | final={final['class_name']} {final['confidence']:.2f} "
            f"| s1={item['class_name']} {item['confidence']:.2f}"
        )
    else:
        label = (
            f"ESC {reason} | final={final['class_name']} {final['confidence']:.2f} "
            f"| s2={best['class_name']} {best['confidence']:.2f} "
            f"| s1={item['class_name']} {item['confidence']:.2f}"
        )

    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage hybrid YOLO detector with conservative relabel logic."
    )
    parser.add_argument(
        "--fast-model", required=True, help="Path to the fast first-stage model."
    )
    parser.add_argument(
        "--slow-model", required=True, help="Path to the second-stage model."
    )
    parser.add_argument(
        "--image-dir",
        default="yolo/images/val",
        help="Directory containing validation images.",
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="Image IDs without file extension."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Escalation confidence threshold."
    )
    parser.add_argument(
        "--min-box-area", type=float, default=0.0, help="Minimum allowed box area."
    )
    parser.add_argument(
        "--min-pad",
        type=int,
        default=32,
        help="Minimum crop padding in pixels for escalated detections.",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.25,
        help="Crop padding as a fraction of max(box_width, box_height).",
    )
    parser.add_argument(
        "--stage1-imgsz",
        type=int,
        default=640,
        help="Inference image size for stage 1.",
    )
    parser.add_argument(
        "--stage2-imgsz",
        type=int,
        default=416,
        help="Inference image size for stage 2 crop re-checks.",
    )
    parser.add_argument(
        "--similar-min-conf",
        type=float,
        default=0.85,
        help="Minimum stage-2 confidence to relabel between similar classes.",
    )
    parser.add_argument(
        "--similar-min-delta",
        type=float,
        default=0.20,
        help="Minimum confidence gain to relabel between similar classes.",
    )
    parser.add_argument(
        "--general-min-conf",
        type=float,
        default=0.75,
        help="Minimum stage-2 confidence for non-similar class relabels.",
    )
    parser.add_argument(
        "--general-min-delta",
        type=float,
        default=0.15,
        help="Minimum confidence gain for non-similar class relabels.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show stage-1 annotated predictions in GUI windows.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save final annotated images with escalations.",
    )
    parser.add_argument(
        "--vis-dir",
        default="runs/hybrid_vis",
        help="Directory for saved visualization images.",
    )
    args = parser.parse_args()

    if args.save_vis:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)

    stage1 = Model(
        args.fast_model,
        min_confidence_threshold=args.threshold,
        min_box_area=args.min_box_area,
        box_enlargement=args.min_pad,
        crop_padding_ratio=args.pad_ratio,
    )
    stage2 = Model(
        args.slow_model,
        min_confidence_threshold=args.threshold,
        min_box_area=0,
        box_enlargement=args.min_pad,
        crop_padding_ratio=args.pad_ratio,
    )

    metrics: Metrics = {
        "images_processed": 0,
        "stage1_detections": 0,
        "escalated": 0,
        "improved": 0,
        "stage2_no_detection": 0,
        "stage2_agree": 0,
        "accepted_relabels": 0,
        "rejected_relabels": 0,
        "kept_stage1_similar": 0,
        "stage1_ms": [],
        "stage2_ms": [],
        "stage1_conf_all": [],
        "stage1_conf_escalated": [],
        "stage2_conf_best": [],
        "confidence_delta": [],
    }

    image_paths = resolve_images(args.image_dir, args.images)
    wall_start = perf_counter()

    for image_path in image_paths:
        print("\n" + "-" * 72)
        print(f"IMAGE: {image_path.name}")
        print("-" * 72)

        if not image_path.exists():
            print("Image not found. Skipping.")
            continue

        metrics["images_processed"] += 1
        stage1.predict(str(image_path), imgsz=args.stage1_imgsz)
        metrics["stage1_ms"].append(stage1.last_inference_ms)

        stage1_detections = stage1.detection_count()
        stage1_confidences = stage1.confidences()
        uncertain = stage1.eval()

        metrics["stage1_detections"] += stage1_detections
        metrics["stage1_conf_all"].extend(stage1_confidences)
        metrics["escalated"] += len(uncertain)

        print(f"Stage-1 inference time: {stage1.last_inference_ms:.2f} ms")
        print(f"Stage-1 detections kept after filtering: {stage1_detections}")
        print(f"Escalated detections: {len(uncertain)}")
        stage1.present(show=args.show)

        vis_image = stage1.annotated_image() if args.save_vis else None

        for item in uncertain:
            original_conf = float(item["confidence"])
            metrics["stage1_conf_escalated"].append(original_conf)

            stage2.predict(item["crop"], imgsz=args.stage2_imgsz)
            metrics["stage2_ms"].append(stage2.last_inference_ms)
            best = stage2.best_detection()

            print(
                f"\nEscalated stage-1 detection #{item['index']} "
                f"({item['class_name']}, conf={original_conf:.4f})"
            )
            print(
                f"Crop box used for stage-2: {item['crop_xyxy']} "
                f"(pad={item['pad_used']})"
            )
            print(f"Stage-2 inference time: {stage2.last_inference_ms:.2f} ms")

            if best is None:
                metrics["stage2_no_detection"] += 1
                final_class, final_conf, reason, _ = choose_final_detection(
                    stage1_class=item["class_name"],
                    stage1_conf=original_conf,
                    best=None,
                    similar_min_conf=args.similar_min_conf,
                    similar_min_delta=args.similar_min_delta,
                    general_min_conf=args.general_min_conf,
                    general_min_delta=args.general_min_delta,
                )
                final = {
                    "class_name": final_class,
                    "confidence": final_conf,
                    "reason": reason,
                }
                print("Stage-2 result: no detection returned.")
                print(
                    f"Final decision: keep stage-1 {final['class_name']} "
                    f"({final['confidence']:.4f}) because {reason}."
                )
                if vis_image is not None:
                    vis_image = draw_stage2_on_image(vis_image, item, None, final)
                continue

            best_conf = float(best["confidence"])
            delta = best_conf - original_conf
            metrics["stage2_conf_best"].append(best_conf)
            metrics["confidence_delta"].append(delta)
            if delta > 0:
                metrics["improved"] += 1

            final_class, final_conf, reason, stage2_class = choose_final_detection(
                stage1_class=item["class_name"],
                stage1_conf=original_conf,
                best=best,
                similar_min_conf=args.similar_min_conf,
                similar_min_delta=args.similar_min_delta,
                general_min_conf=args.general_min_conf,
                general_min_delta=args.general_min_delta,
            )

            if reason == "agree":
                metrics["stage2_agree"] += 1
            elif reason in {"relabel", "similar_class_relabel"}:
                metrics["accepted_relabels"] += 1
            else:
                metrics["rejected_relabels"] += 1
                if reason == "kept_stage1_similar":
                    metrics["kept_stage1_similar"] += 1

            final = {
                "class_name": final_class,
                "confidence": final_conf,
                "reason": reason,
            }

            print(
                f"Stage-2 best detection: {best['class_name']} "
                f"(conf={best_conf:.4f}, delta={delta:+.4f})"
            )
            print(
                f"Final decision: {final_class} ({final_conf:.4f}) | "
                f"reason={reason} | stage1={item['class_name']} -> stage2={stage2_class}"
            )
            if vis_image is not None:
                vis_image = draw_stage2_on_image(vis_image, item, best, final)

        if vis_image is not None:
            out_path = Path(args.vis_dir) / f"annotated_{image_path.name}"
            cv2.imwrite(str(out_path), vis_image)
            print(f"Saved visualization: {out_path}")

    total_wall_ms = (perf_counter() - wall_start) * 1000.0
    print_summary(metrics, total_wall_ms)


if __name__ == "__main__":
    main()
