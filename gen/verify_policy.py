"""Independent Verifier for Enterprise Policy & Tool Routing Benchmark.

Re-parses the rendered natural language state, re-constructs the facts and rules,
executes independent policy verification logic, and validates zero label noise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def verify_policy_sample(sample: dict) -> bool:
    state = sample["state"]
    q1 = sample["questions"]["q1"]
    expected_label = q1["label"]
    
    # 1. Parse Actor info
    m_role = re.search(r"Role: (\w+)", state)
    m_clr = re.search(r"Clearance: Level (\d+)", state)
    m_net = re.search(r"Ingress Network: (\w+)", state)
    m_mfa = re.search(r"MFA_Active=(True|False)", state)
    m_mgr = re.search(r"Manager_Approved=(True|False)", state)
    
    if not (m_role and m_clr and m_net and m_mfa and m_mgr):
        return False
        
    role = m_role.group(1)
    clr = int(m_clr.group(1))
    net = m_net.group(1)
    mfa = m_mfa.group(1) == "True"
    mgr = m_mgr.group(1) == "True"
    
    # 2. Parse Rules
    m_req_clr = re.search(r"requires minimum clearance Level (\d+)", state)
    req_clr = int(m_req_clr.group(1)) if m_req_clr else 1
    
    has_rule_d = "Rule D:" in state
    
    # Rule evaluations
    clr_ok = clr >= req_clr
    net_ok = mfa if net in ("external_public", "corp_vpn") else True
    mgr_ok = mgr
    
    # Check depth based on questions
    ins = q1["instructions"]
    if "Rule A," in ins or "Rule A:" in ins or "only on Rule A" in ins or "Rule A," in ins or "context and Rule A" in ins:
        k = 1
    elif "Rules A and B" in ins:
        k = 2
    elif "Rules (A, B, and C)" in ins or "Rules A, B, and C" in ins:
        k = 3
    else:
        k = 4
        
    # Decision logic
    if k == 1:
        computed = "route_to_direct_execution" if clr_ok else "route_to_security_deny"
    elif k == 2:
        if not clr_ok:
            computed = "route_to_security_deny"
        elif not net_ok:
            computed = "route_to_mfa_challenge"
        else:
            computed = "route_to_direct_execution"
    elif k == 3:
        if not clr_ok:
            computed = "route_to_security_deny"
        elif not net_ok:
            computed = "route_to_mfa_challenge"
        elif not mgr_ok:
            computed = "route_to_approval_queue"
        else:
            computed = "route_to_direct_execution"
    else:
        is_sandbox_eligible = (role in ["developer", "analyst"])
        if not clr_ok:
            computed = "route_to_sandbox_runner" if is_sandbox_eligible and clr >= 1 else "route_to_security_deny"
        elif not net_ok:
            computed = "route_to_mfa_challenge"
        elif not mgr_ok:
            computed = "route_to_sandbox_runner" if is_sandbox_eligible else "route_to_approval_queue"
        else:
            computed = "route_to_direct_execution"
            
    return computed == expected_label


def main():
    paths = sorted(Path("data").glob("policy_*.jsonl"))
    if not paths:
        print("No policy_*.jsonl files found in data/")
        sys.exit(1)
        
    total_samples = 0
    total_fails = 0
    for p in paths:
        samples = [json.loads(line) for line in open(p)]
        fails = 0
        for idx, s in enumerate(samples):
            if not verify_policy_sample(s):
                fails += 1
        total_samples += len(samples)
        total_fails += fails
        status = "FAILS=NONE" if fails == 0 else f"FAILS={fails}"
        print(f"{p.name:25s} samples={len(samples):<4d} {status}")
        
    print(f"\nPolicy Verification complete: {total_samples} samples checked, {total_fails} errors.")
    if total_fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
