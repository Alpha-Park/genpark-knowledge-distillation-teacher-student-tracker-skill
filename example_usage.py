"""
Demonstration of Knowledge Distillation Teacher Student Tracker Skill
"""

from client import KnowledgeDistillationTracker

def main():
    print("=== Evaluating Knowledge Distillation Loss ===")
    distiller = KnowledgeDistillationTracker(temperature=2.5, alpha=0.6)

    teacher_logits = [2.8, 1.2, -0.5, 0.1]
    student_logits = [2.0, 0.9, -0.2, 0.3]
    true_label = 0

    metrics = distiller.evaluate_distillation_step(student_logits, teacher_logits, true_label)
    print("Distillation Step Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    assert metrics["total_loss"] > 0.0
    print("\nKnowledge Distillation Verification PASS!")

if __name__ == "__main__":
    main()
