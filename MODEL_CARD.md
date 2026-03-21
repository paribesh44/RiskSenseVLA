# RiskSense-VLA Model Card

## Model Details

- **Name**: RiskSense-VLA
- **Type**: Hazard-aware Vision-Language-Action (VLA) system
- **Description**: A predictive VLA system designed to detect and reason about hazards in human-robot interaction scenarios. Combines perception (object detection), human-object interaction (HOI) prediction, and hazard reasoning modules.

## Intended Use

- **Primary**: Safety-critical environments where robots operate near humans
- **Primary**: Robotic manipulation in shared workspaces
- **Primary**: Human-robot collaboration scenarios requiring hazard awareness
- **Secondary**: Research and benchmarking of hazard-aware perception systems

## Out-of-Scope Use

- **Not for autonomous driving**: The model is not trained or validated for road or vehicle scenarios.
- **Not for medical decisions**: Do not use for clinical diagnosis, treatment, or medical device control.
- **Not for high-stakes autonomous decision-making**: Intended as an assistive component; human oversight is recommended.

## Training Data

- **Source**: Synthetic procedural hazard scenes
- **Generation**: Procedural rendering and optional Stable Diffusion-based augmentation
- **Content**: Simulated human-robot interaction scenarios with annotated hazards, HOIs, and temporal sequences
- **Limitation**: No real-world imagery used in training

## Evaluation Metrics

- **THC** (Temporal HOI Consistency): Proportion of consecutive frames where the top observed HOI action remains unchanged
- **HAA** (Hazard-Aware Accuracy): Accuracy of hazard detection and scoring
- **RME** (Reactive Memory Efficiency): Efficiency of hazard memory and retrieval
- **FPS**: Inference throughput (frames per second)

## Ethical Considerations

- **False negatives in hazard detection**: Missed hazards can lead to unsafe robot behavior. Mitigate through conservative thresholds and human-in-the-loop validation.
- **Bias in training data**: Synthetic data may not represent all demographics, environments, or hazard types. Real-world validation is recommended before deployment.
- **Transparency**: Log hazard scores and HOI predictions for auditability.

## Limitations

- **Synthetic-only training**: No real-world validation yet; performance on real imagery is unknown.
- **Single-camera only**: No multi-view or stereo fusion.
- **Domain gap**: Procedural/synthetic scenes may not generalize to natural environments.
- **Temporal scope**: Limited to short video clips; long-horizon reasoning is not evaluated.

## Risks and Mitigations


| Risk                              | Mitigation                                                                |
| --------------------------------- | ------------------------------------------------------------------------- |
| False negatives (missed hazards)  | Use conservative hazard thresholds; combine with rule-based safety layers |
| Over-reliance on synthetic data   | Validate on real-world benchmarks before deployment                       |
| Latency in safety-critical loops  | Profile FPS; consider quantization (int8) for faster inference            |
| Model drift or distribution shift | Periodic retraining and evaluation on held-out data                       |


