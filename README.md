# GenPark Knowledge Distillation Teacher Student Tracker Skill

Teacher-student knowledge distillation loss engine with temperature-scaled soft targets and KL divergence.

Explore more frameworks at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    A[Teacher Model] -->|Logits / T| B[Soft Targets Q]
    C[Student Model] -->|Logits / T| D[Soft Predictions P]
    B & D --> E[Distillation Loss = T^2 * KL(Q || P)]
    C -->|Logits / 1.0| F[Hard Cross-Entropy Loss]
    E & F --> G[Total L = (1-alpha)*L_hard + alpha*L_distill]
```

## Features
- Temperature scaling parameterization.
- Analytical KL divergence computation.
- Pure Python standard library.
