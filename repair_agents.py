import re

# Base Agent

class BaseRepairAgent:
    def repair_text(self, S, prompt):
        raise NotImplementedError


# 1️ Filler Removal Agent

class FillerAgent(BaseRepairAgent):
    def repair_text(self, S, prompt):
        # Match fillers ignoring surrounding commas/formatting
        fillers = r"\b(uh|um|ah|er|like|hmm|you know)\b[,\.\s]*"
        return re.sub(fillers, "", S, flags=re.IGNORECASE).strip(" ,.")


# 2️ Repetition Repair Agent

class RepetitionAgent(BaseRepairAgent):
    def repair_text(self, S, prompt):
        # 1. Repeated full words (handling Whisper's commas: "I, I, I went" -> "I went")
        S = re.sub(r"\b(\w+)(?:[,\.\s]+\1\b)+", r"\1", S, flags=re.IGNORECASE)

        # 2. Hyphenated partials: "b-b-boy" -> "boy"
        S = re.sub(r"\b(?:\w+-)+\w+\b", lambda m: m.group(0).split('-')[-1], S)
        
        # 3. Spaced prefix stutters (handling commas: "st, st, store" -> "store")
        S = re.sub(r"\b([a-zA-Z]{1,3})[,\.\s]+(?=\1[a-zA-Z]+)", "", S, flags=re.IGNORECASE)

        # Clean lingering duplicated commas and spaces
        S = re.sub(r"([,\.])\1+", r"\1", S) 
        S = re.sub(r"\s+", " ", S)

        return S.strip(" ,.")


# 3️ Prolongation Agent

class ProlongationAgent(BaseRepairAgent):
    def repair_text(self, S, prompt):
        # Reduce excessive character repetition (Ssssschool → School)
        S = re.sub(r"([a-zA-Z])\1{2,}", r"\1", S)
        return S.strip(" ,.")


# Lightweight Grammar Agent

class GrammarAgent(BaseRepairAgent):
    def repair_text(self, S, prompt):
        if not S:
            return S

        # 1. Clean spacing around punctuation
        S = re.sub(r"\s+([,\.\!\?])", r"\1", S)

        # 2. Capitalize standalone 'i'
        S = re.sub(r"\bi\b", "I", S)

        # 3. Capitalize first letter of every sentence
        def cap_match(m):
            return m.group(1) + m.group(2).upper()
        S = re.sub(r"([\.\!\?]\s+)([a-z])", cap_match, S)

        S = " ".join(S.split())
        S = S.strip(" ,.-")
        
        if not S:
            return S

        S = S[0].upper() + S[1:]
        if not S.endswith((".", "!", "?")):
            S += "."

        return S
