# Case Sensitivity in NLP Parsing

## Problem Statement

Facts stored with one casing couldn't be retrieved with different casing:
```
Store: "Alice's job is Engineer"
Query: "What is alice's job?"
Result: "I don't know" ← Should return "Engineer"
```

## Root Cause Analysis

### The Bug

The parser extracted entities preserving original case, but queries used different case:

```python
# parser.py - extracted "Alice"
def _extract_possessive(self, match):
    return {"entity": match.group(1), ...}  # "Alice"

# reasoner.py - no normalization
def query(self, entity, role):
    result = self.ai.query_fact(entity, role)  # Looks for "alice"
```

The HippocampusVSA stores concepts case-sensitively:
```python
item_memory["Alice"] = hv1
item_memory["alice"] = hv2  # Different entry!
```

## Solution

### Normalize at storage and retrieval boundaries:

```python
# reasoner.py
def store_fact(self, entity, role, value):
    entity = entity.lower()
    role = role.lower()
    value = value.lower()
    # ...store...

def query(self, entity, role):
    entity = entity.lower()
    role = role.lower()
    # ...query...
```

### Why at reasoner level?

- Parser preserves original for display purposes
- Core AI handles normalized form
- Single normalization point = fewer bugs

## Verification Method

```bash
python3 test_intelligence.py
```

### Test Cases

```python
# Test: Case-insensitive retrieval
agent.chat("Alice's job is Engineer")
response = agent.chat("What is alice's job?")
assert "engineer" in response.text.lower()

# Test: Mixed case storage and retrieval
agent.chat("BOB works at GOOGLE")
response = agent.chat("where does bob work?")
assert "google" in response.text.lower()
```

## Test Results

| Test | Before | After |
|------|--------|-------|
| Same case retrieval | PASS | PASS |
| Different case retrieval | FAIL | PASS |
| Mixed case storage | FAIL | PASS |

### Intelligence Score Impact

- Understanding: 86% → 100%
- Overall: 90% → 100%

## Code Changes

**File:** `src/conversation/reasoner.py`

```python
def query(self, entity: str, role: str, follow_chain: bool = True):
    # ADD: Normalize case
    entity = entity.lower()
    role = role.lower()
    # ... rest of method

def store_fact(self, entity: str, role: str, value: str):
    # ADD: Normalize case
    entity = entity.lower()
    role = role.lower()
    value = value.lower()
    # ... rest of method
```

## Lessons Learned

1. **Normalize at boundaries** - Not inside core storage
2. **Test with variations** - Same logic, different inputs
3. **Preserve for display** - Parser keeps original for natural responses

## Related

- Text processing best practices
- Unicode normalization (future consideration)

---

*Fixed: 2026-02-16*
*Status: SOLVED*
