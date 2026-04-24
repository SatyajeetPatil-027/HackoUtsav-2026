## Repair Agents

The system uses 4 repair agents that work in parallel to improve speech-to-text output.

### 1. Meaning Preservation Agent
Removes basic fillers (uh, um, etc.) while keeping the original meaning unchanged.

### 2. Grammar-Optimized Agent
Fixes grammar, capitalization, punctuation, and sentence structure.

### 3. Concise-Fluent Agent
Removes repeated words, fillers, and stretched characters to make text fluent.

### 4. Sequence-Optimized Agent
Fixes stutter patterns and improves word sequence for better flow.

---

## Approach

All agents process the same input independently (parallel approach).  
Their outputs are then evaluated by the **Master Agent**, which selects the best result using a scoring system.
