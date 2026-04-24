import re


# =========================
# Base Agent
# =========================

class BaseRepairAgent:
    def repair_text(self, S, prompt=None):
        raise NotImplementedError


# =========================
# 1. Meaning Preservation Agent
# Goal: Minimal changes, keep original meaning safe
# =========================

class MeaningPreservationAgent(BaseRepairAgent):
    def repair_text(self, S, prompt=None):
        if not S:
            return S

        # Only remove obvious noise (safe cleaning)
        S = re.sub(r"\b(uh|um|ah|er|hmm|you know)\b[,\.\s]*", "", S, flags=re.IGNORECASE)

        # Fix spacing
        S = re.sub(r"\s+", " ", S)

        return S.strip(" ,.")


# =========================
# 2. Grammar-Optimized Agent
# Goal: Improve grammar + readability
# =========================

class GrammarOptimizedAgent(BaseRepairAgent):
    def repair_text(self, S, prompt=None):
        if not S:
            return S

        # Fix spacing before punctuation
        S = re.sub(r"\s+([,\.\!\?])", r"\1", S)

        # Capitalize "i"
        S = re.sub(r"\bi\b", "I", S)

        # Capitalize sentence starts
        def cap(m):
            return m.group(1) + m.group(2).upper()

        S = re.sub(r"([\.\!\?]\s+)([a-z])", cap, S)

        # Normalize spaces
        S = " ".join(S.split())

        # Capitalize first letter
        S = S.strip()
        if S:
            S = S[0].upper() + S[1:]

        # Add punctuation if missing
        if S and not S.endswith((".", "!", "?")):
            S += "."

        return S


# =========================
# 3. Concise-Fluent Agent
# Goal: Remove redundancy + improve fluency
# =========================

class ConciseFluentAgent(BaseRepairAgent):
    def repair_text(self, S, prompt=None):
        if not S:
            return S

        # Remove repeated words
        S = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", S, flags=re.IGNORECASE)

        # Remove filler phrases
        S = re.sub(r"\b(uh|um|ah|er|hmm|you know|basically|actually)\b", "", S, flags=re.IGNORECASE)

        # Reduce elongation
        S = re.sub(r"([a-zA-Z])\1{2,}", r"\1", S)

        # Normalize spacing
        S = re.sub(r"\s+", " ", S)

        return S.strip(" ,.")


# =========================
# 4. Sequence-Optimized Agent
# Goal: Improve sentence flow/order (light restructuring)
# =========================

class SequenceOptimizedAgent(BaseRepairAgent):
    def repair_text(self, S, prompt=None):
        if not S:
            return S

        # Remove repeated fragments (simple sequence cleanup)
        S = re.sub(r"\b(\w+)(?:[,\.\s]+\1\b)+", r"\1", S, flags=re.IGNORECASE)

        # Fix stutter patterns
        S = re.sub(r"\b([a-zA-Z]{1,3})[,\.\s]+(?=\1[a-zA-Z]+)", "", S, flags=re.IGNORECASE)

        # Normalize spacing
        S = re.sub(r"\s+", " ", S)

        return S.strip(" ,.")
