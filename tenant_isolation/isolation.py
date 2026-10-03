"""Per-tenant resources and the access check that proves no tenant can reach another tenant's data.

A multi-tenant platform gives each client its own service, image repository, bucket and secrets. The isolation lives in
the IAM policies attached to each tenant's task role. This script generates those policies from one tenant list, the
way a Terraform module with `for_each` would, then evaluates every role against every resource, with a deliberately
small evaluator (Allow statements, `*` wildcards in resource ARNs; no Deny, no conditions).

Two versions of the policies:
  wildcard  buckets and keys scoped per tenant, secrets readable with Resource "*" (the gap we found in an audit)
  scoped   one role per tenant, each statement scoped to that tenant's ARNs

  python tenant_isolation/isolation.py
"""
from __future__ import annotations

import fnmatch
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ACCOUNT, REGION = "111122223333", "us-east-1"
TENANTS = ["tenant-a", "tenant-b", "tenant-c", "tenant-d"]


def resources(tenant):
    return {
        "bucket": f"arn:aws:s3:::platform-{tenant}-data/*",
        "secret_llm_key": f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:platform/{tenant}/llm-key-AbCdEf",
        "secret_db_url": f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:platform/{tenant}/db-url-GhIjKl",
        "kms_key": f"arn:aws:kms:{REGION}:{ACCOUNT}:key/{tenant}-key",
    }


ACTIONS = {"bucket": "s3:GetObject", "secret_llm_key": "secretsmanager:GetSecretValue",
           "secret_db_url": "secretsmanager:GetSecretValue", "kms_key": "kms:Decrypt"}


def shared_policies():
    """Buckets and keys scoped per tenant, but secrets readable with Resource "*": one line, every tenant's secrets."""
    out = {}
    for t in TENANTS:
        r = resources(t)
        out[t] = {"Statement": [
            {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": r["bucket"]},
            {"Effect": "Allow", "Action": ["kms:Decrypt"], "Resource": r["kms_key"]},
        ]}
    return out


def scoped_policies():
    out = {}
    for t in TENANTS:
        r = resources(t)
        out[t] = {"Statement": [
            {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"],
             "Resource": f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:platform/{t}/*"},
            {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": r["bucket"]},
            {"Effect": "Allow", "Action": ["kms:Decrypt"], "Resource": r["kms_key"]},
        ]}
    return out


def allowed(policy, action, arn):
    for s in policy["Statement"]:
        acts = s["Action"] if isinstance(s["Action"], list) else [s["Action"]]
        res = s["Resource"] if isinstance(s["Resource"], list) else [s["Resource"]]
        if s["Effect"] == "Allow" and action in acts and any(fnmatch.fnmatchcase(arn, p) for p in res):
            return True
    return False


def matrix(policies):
    """rows: the role of each tenant; columns: the resources of each tenant; value: resources reachable (of 4)."""
    m = []
    for role in TENANTS:
        row = []
        for owner in TENANTS:
            row.append(sum(allowed(policies[role], ACTIONS[k], arn) for k, arn in resources(owner).items()))
        m.append(row)
    return m


def lint(policies):
    findings = []
    for t, p in policies.items():
        for s in p["Statement"]:
            res = s["Resource"] if isinstance(s["Resource"], list) else [s["Resource"]]
            for r in res:
                if r == "*" or (t not in r and "*" in r):
                    findings.append({"tenant_role": t, "action": s["Action"], "resource": r})
    return findings


if __name__ == "__main__":
    out = {}
    for name, fn in (("wildcard", shared_policies), ("scoped", scoped_policies)):
        pol = fn()
        m = matrix(pol)
        cross = sum(v for i, row in enumerate(m) for j, v in enumerate(row) if i != j)
        out[name] = {"matrix": m, "cross_tenant_grants": cross, "lint_findings": len(lint(pol)),
                     "examples": lint(pol)[:3]}
        print(name, "cross-tenant grants:", cross, "lint findings:", len(lint(pol)))
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
