"""
Knowledge Distillation Teacher Student Tracker Skill Client
Pure Python Standard Library implementation of Knowledge Distillation (Hinton et al.).
Calculates temperature-scaled soft targets, student-teacher Kullback-Leibler (KL) divergence,
and combined task-distillation loss for agent capability retention.
"""

from typing import List, Dict, Any, Tuple, Optional
import math


class KnowledgeDistillationTracker:
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss vs hard label loss

    @staticmethod
    def softmax_with_temperature(logits: List[float], temp: float) -> List[float]:
        scaled = [x / temp for x in logits]
        max_v = max(scaled)
        exp_v = [math.exp(x - max_v) for x in scaled]
        sum_exp = sum(exp_v)
        return [x / sum_exp for x in exp_v]

    def compute_kl_divergence(self, p_student: List[float], q_teacher: List[float]) -> float:
        """KL(Q || P) = sum q_i * log(q_i / p_i)."""
        kl = 0.0
        for p, q in zip(p_student, q_teacher):
            if q > 1e-12 and p > 1e-12:
                kl += q * math.log(q / p)
        return kl

    def evaluate_distillation_step(self, student_logits: List[float], teacher_logits: List[float],
                                   true_class_idx: int) -> Dict[str, float]:
        # Soft probabilities at temperature T
        p_soft_student = self.softmax_with_temperature(student_logits, self.temperature)
        q_soft_teacher = self.softmax_with_temperature(teacher_logits, self.temperature)

        # Distillation loss (KL divergence * T^2 per Hinton et al.)
        kl_loss = self.compute_kl_divergence(p_soft_student, q_soft_teacher)
        scaled_distill_loss = kl_loss * (self.temperature ** 2)

        # Hard label cross-entropy loss (at T=1)
        p_hard_student = self.softmax_with_temperature(student_logits, 1.0)
        hard_loss = -math.log(max(p_hard_student[true_class_idx], 1e-12))

        # Total combined loss
        total_loss = (1.0 - self.alpha) * hard_loss + self.alpha * scaled_distill_loss

        return {
            "total_loss": total_loss,
            "distillation_loss": scaled_distill_loss,
            "hard_label_loss": hard_loss,
            "student_top_prob": max(p_hard_student),
            "teacher_soft_entropy": -sum(q * math.log(max(q, 1e-12)) for q in q_soft_teacher)
        }
