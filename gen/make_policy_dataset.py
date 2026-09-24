"""Enterprise Policy & Tool Routing Benchmark Generator.

Simulates multi-hop access control and agent tool routing policies:
- User identity, role, and clearance
- Network context and authentication requirements
- Resource sensitivity and regulatory constraints
- Cascading decision rules (k-hop chains) -> Destination Tool Action

Serves as the Application Benchmark complementing the Spatial Diagnostic Benchmark,
demonstrating System 1.5 viability in realistic agent architectures.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OUT = Path("data")

ROLES = ["analyst", "developer", "auditor", "operator", "contractor", "manager"]
RESOURCES = ["sales_log", "financial_audit", "user_pii", "source_repo", "prod_database", "billing_archive"]
NETWORKS = ["internal_office", "corp_vpn", "external_public", "cloud_bastion"]
ACTIONS = [
    "route_to_approval_queue",
    "route_to_mfa_challenge",
    "route_to_direct_execution",
    "route_to_sandbox_runner",
    "route_to_security_deny"
]

ROUTING_CRITERIA = {
    "route_to_approval_queue": "Escalate request to compliance or manager review queue",
    "route_to_mfa_challenge": "Redirect user to multi-factor authentication challenge",
    "route_to_direct_execution": "Grant immediate execution and dispatch to worker service",
    "route_to_sandbox_runner": "Execute action in isolated read-only sandbox environment",
    "route_to_security_deny": "Immediately block request and log security violation"
}

RULE_TEMPLATES = [
    "- Policy Rule {id}: If actor role is '{role}' accessing '{res}', clearance level must be at least {lvl}.",
    "- Policy Rule {id}: Accessing '{res}' from network '{net}' requires {sec_req}.",
    "- Policy Rule {id}: Operation on '{res}' with sensitivity '{sens}' mandates {appr_req}.",
    "- Policy Rule {id}: If network context is '{net}' and role is '{role}', execution mode defaults to '{mode}'."
]


def gen_policy_sample(rng: random.Random, k: int, sample_id: int = 1) -> dict:
    """Generate a k-hop policy evaluation request."""
    user = f"user_{rng.randint(100, 999)}"
    role = rng.choice(ROLES)
    res = rng.choice(RESOURCES)
    net = rng.choice(NETWORKS)
    user_clearance = rng.randint(1, 4)
    has_mfa = rng.choice([True, False])
    has_manager_approval = rng.choice([True, False])
    
    # Generate atomic facts
    context_lines = [
        f"Incoming Request Context for {user}:",
        f"- Actor: {user}, Role: {role}, Clearance: Level {user_clearance}.",
        f"- Target Resource: {res}.",
        f"- Ingress Network: {net}.",
        f"- Security Flags: MFA_Active={has_mfa}, Manager_Approved={has_manager_approval}."
    ]
    
    policy_lines = ["Active Policy Rules:"]
    # Chain conditions
    # Rule 1: Clearance check
    req_clearance = rng.randint(1, 3)
    policy_lines.append(f"- Rule A: Accessing '{res}' by '{role}' requires minimum clearance Level {req_clearance}.")
    clearance_ok = user_clearance >= req_clearance
    
    # Rule 2: Network / MFA check
    policy_lines.append(f"- Rule B: Requests from '{net}' for '{res}' require MFA_Active=True to bypass challenge.")
    network_ok = has_mfa if net in ("external_public", "corp_vpn") else True
    
    # Rule 3: Approval check
    policy_lines.append(f"- Rule C: High-impact actions on '{res}' require Manager_Approved=True for direct execution.")
    approval_ok = has_manager_approval
    
    # Cascade decision based on k
    if k == 1:
        # 1-hop: Depends only on Rule A
        label = "route_to_direct_execution" if clearance_ok else "route_to_security_deny"
        instruction = f"Based on the context and Rule A, what is the valid routing destination for {user}?"
    elif k == 2:
        # 2-hop: Clearance + Network
        if not clearance_ok:
            label = "route_to_security_deny"
        elif not network_ok:
            label = "route_to_mfa_challenge"
        else:
            label = "route_to_direct_execution"
        instruction = f"Evaluating Rules A and B simultaneously, what is the final routing destination for {user}?"
    elif k == 3:
        # 3-hop: Clearance + Network + Approval
        if not clearance_ok:
            label = "route_to_security_deny"
        elif not network_ok:
            label = "route_to_mfa_challenge"
        elif not approval_ok:
            label = "route_to_approval_queue"
        else:
            label = "route_to_direct_execution"
        instruction = f"Considering all Policy Rules (A, B, and C), what is the definitive routing destination for {user}?"
    else:
        # k == 4: Extra sandbox condition
        is_sandbox_eligible = (role in ["developer", "analyst"])
        policy_lines.append(f"- Rule D: Blocked or unapproved requests for '{role}' default to sandbox if clearance >= 1.")
        if not clearance_ok:
            label = "route_to_sandbox_runner" if is_sandbox_eligible and user_clearance >= 1 else "route_to_security_deny"
        elif not network_ok:
            label = "route_to_mfa_challenge"
        elif not approval_ok:
            label = "route_to_sandbox_runner" if is_sandbox_eligible else "route_to_approval_queue"
        else:
            label = "route_to_direct_execution"
        instruction = f"Executing full cascading evaluation of Rules A, B, C, and D, determine the target destination tool for {user}:"

    # Assemble state
    state = "\n".join(context_lines) + "\n\n" + "\n".join(policy_lines)
    
    question = {
        "type": "choice",
        "instructions": instruction,
        "criteria": ROUTING_CRITERIA,
        "label": label
    }
    
    return {
        "state": state,
        "questions": {"q1": question},
        "meta": {"k": k, "user": user, "target": label}
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()
    
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)
    
    # 1. Training set: k=1 (250) + k=2 (250) = 500 samples
    train_samples = []
    for k in [1, 2]:
        for _ in range(250):
            train_samples.append(gen_policy_sample(rng, k))
    rng.shuffle(train_samples)
    
    clean_train = [{"state": s["state"], "questions": s["questions"]} for s in train_samples]
    with open(OUT / "policy_train_k12.jsonl", "w") as f:
        for s in clean_train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(clean_train)} samples in data/policy_train_k12.jsonl")
    
    # 2. Test sets: k=1..4 (100 each)
    for k in range(1, 5):
        test_samples = [gen_policy_sample(rng, k) for _ in range(100)]
        clean_test = [{"state": s["state"], "questions": s["questions"]} for s in test_samples]
        with open(OUT / f"policy_test_k{k}.jsonl", "w") as f:
            for s in clean_test:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Generated 100 samples in data/policy_test_k{k}.jsonl")


if __name__ == "__main__":
    main()
