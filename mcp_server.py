"""
MCP Server for Knowledge Distillation Teacher Student Tracker Skill
"""

import json
import sys
from client import KnowledgeDistillationTracker

dist = KnowledgeDistillationTracker()

def handle_call(name: str, args: dict) -> dict:
    if name == "compute_distillation":
        s_logits = args.get("student_logits", [])
        t_logits = args.get("teacher_logits", [])
        label = args.get("target_label", 0)
        return dist.evaluate_distillation_step(s_logits, t_logits, label)
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
