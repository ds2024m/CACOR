class PromptTemplates:
    SYSTEM_PROMPT = "You are an emotion recognition expert. You must follow the output format exactly: first 'Explanation: ...', then 'Emotion: <one of anger,disgust,fear,joy,neutral,sadness,surprise>', then 'Confidence: <0.0-1.0>', and finally the tag '<final>Conclusion: the dominant emotion is EMOTION.</final>'. Do not add extra text."

    @staticmethod
    def _build_context(utterance: str, audio_analysis: str) -> str:
        ctx = ""
        if utterance:
            ctx += f'\nDIALOGUE: "{utterance}"\n'
        if audio_analysis:
            ctx += f'AUDIO_ACOUSTICS: {audio_analysis}\n'
        if ctx:
            ctx += "Consider the multimodal context provided above.\n"
        return ctx

    @staticmethod
    def visual_agent(utterance: str, audio_analysis: str,
                     timestamps: Optional[List[float]] = None,
                     video_duration: Optional[float] = None,
                     compare_frames: bool = True) -> str:
        temporal = ""
        if timestamps and video_duration and len(timestamps) > 0:
            frames_desc = ", ".join([f"Frame {i+1} (t={t:.1f}s)" for i, t in enumerate(timestamps)])
            temporal = (f"Video segment duration: {video_duration:.1f}s. "
                        f"Frames captured at: {frames_desc}. ")
            if compare_frames:
                temporal += "Explicitly compare the expressions across frames: note how they change over time. "
        return f"""
{PromptTemplates.SYSTEM_PROMPT}
You are a visual emotion expert. FOCUS ON VISUAL CUES: facial expressions, body language, and their temporal evolution.
{temporal}
DIALOGUE: "{utterance}"
Output format (REQUIRED):
Explanation: <2-3 sentences, comparing frames if possible>
Emotion: <ONE OF: {', '.join(EMOTION_NAMES)}>
Confidence: <number between 0.0 and 1.0>
{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}"""

    @staticmethod
    def speech_agent(utterance: str, audio_analysis: str, audio_emotion: str = "unknown",
                     audio_confidence: float = 0.0) -> str:
        classifier_info = f"EXTERNAL ACOUSTIC CLASSIFIER says emotion = {audio_emotion} with confidence {audio_confidence:.2f}. "
        if audio_confidence > 0.7:
            classifier_info += "You must strongly consider this prediction. Only override if the dialogue content provides clear contradictory evidence."
        else:
            classifier_info += "You may use this as a hint, but base your decision primarily on the dialogue and acoustic details."
        return f"""
{PromptTemplates.SYSTEM_PROMPT}
You are a speech and pragmatic emotion expert. Analyze the alignment between words and acoustic delivery.
DIALOGUE: "{utterance}"
{classifier_info}
ACOUSTIC DETAILS: {audio_analysis}
Output format (REQUIRED):
Explanation: <2-3 sentences>
Emotion: <ONE OF: {', '.join(EMOTION_NAMES)}>
Confidence: <number between 0.0 and 1.0>
{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}"""

    @staticmethod
    def synthesizer(visual_result: str, speech_result: str, utterance: str,
                    audio_analysis: str, audio_emotion: str = "unknown",
                    audio_confidence: float = 0.0, policies: Set[str] = None) -> str:
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        # Build policies section dynamically based on enabled policies
        policies_text = ""
        if policies and len(policies) > 0:
            policies_text = "\n=== RESOLUTION POLICIES (MUST FOLLOW STRICTLY) ===\n"
            policies_text += "You are given a Visual Agent report and a Speech Agent report.\n"
            policies_text += "First, map any nuanced emotions to the 7 allowed classes:\n"
            policies_text += "  - Frustration / Annoyance / Hostility -> ANGER or DISGUST\n"
            policies_text += "  - Amusement / Relief -> JOY\n"
            policies_text += "  - Concern / Anxiety / Discomfort -> FEAR or SADNESS\n"
            policies_text += "  - Sarcasm / Irony -> ANGER or DISGUST\n\n"
            policies_text += "Then apply the following priority rules in order:\n"
            if "A" in policies:
                policies_text += "[A] VISUAL DOMINANCE: If Visual Agent confidence ≥ 0.8 and Speech Agent confidence < 0.6, trust Visual.\n"
            if "B" in policies:
                policies_text += "[B] SPEECH DOMINANCE: If Speech Agent confidence ≥ 0.8 and Visual Agent confidence < 0.6, trust Speech.\n"
            if "C" in policies:
                policies_text += "[C] CONSENSUS: If both agents agree on the same emotion, output that emotion (regardless of confidence).\n"
            if "D" in policies:
                policies_text += "[D] AMBIGUOUS CONFLICT: If the agents disagree and rules A, B, C do not apply, check for this pattern: Visual expresses a POSITIVE emotion (Joy, Surprise) AND Speech expresses a NEGATIVE emotion (Anger, Disgust, Sadness) OR the dialogue contains obvious verbal irony (e.g., Oh great) then PRIORITIZE SPEECH. If the emotions disagree and one of the emotions are neutral, choose neutral. If confidence difference are smaller than 0.2 then trust the agent with HIGHER CONFIDENCE. If the confidence are equal (eg 0.5 and 0.5) OR CLOSE (difference smaller than 0.2) then TRUST VISUAL.\n"                         
                policies_text += "\nYou MUST output exactly ONE emotion from: anger, disgust, fear, joy, neutral, sadness, surprise.\n"

            policies_text += "In your Explanation, explicitly mention which policy(s) you applied and why.\n"
        else:
            policies_text = "\n=== NO EXPLICIT POLICIES ===\nSynthesize the two reports using your own judgment. Output exactly one emotion from the allowed list.\n"

        return f"""{ctx}
{policies_text}
You are the Lead Synthesis Expert. You have NO DIRECT ACCESS to video.
=== VISUAL AGENT REPORT ===
{visual_result}
=== SPEECH AGENT REPORT ===
{speech_result}
Output format (REQUIRED):
Explanation: <2-3 sentences explaining your reasoning>
Emotion: <ONE OF: {', '.join(EMOTION_NAMES)}>
Confidence: <number between 0.0 and 1.0>
{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}"""

    @staticmethod
    def direct(utterance: str, audio_analysis: str) -> str:
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        return f"""{ctx}
{PromptTemplates.SYSTEM_PROMPT}
Analyze the video frames and classify the dominant emotion expressed.
Summarize your reasoning with few words.
Choose ONE from: {', '.join(EMOTION_NAMES)}.
{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}"""

    @staticmethod
    def chain_of_thought(utterance: str, audio_analysis: str) -> str:
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        return f"""{ctx}
{PromptTemplates.SYSTEM_PROMPT}
Analyze the video frames step-by-step:
1. VISUAL: Facial expressions and body language across the video sequence
2. SPEECH: Dialogue content and vocal prosody
3. SYNTHESIS: Integrate visual cues with speech information
Summarize your reasoning with few words.
Choose ONE from: {', '.join(EMOTION_NAMES)}.
{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}"""

    @staticmethod
    def chain_of_thought_turn1(utterance: str, audio_analysis: str) -> str:
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        return f"""{ctx}
STEP-BY-STEP ANALYSIS (Part 1):
1. What facial expressions do you observe across the video? Compare frames if multiple.
2. What body language cues are present?
3. What does the speech content and tone suggest?
Summarize your reasoning with few words."""

    @staticmethod
    def chain_of_thought_turn2(utterance: str, audio_analysis: str, turn1_response: str) -> str:
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        return f"""{ctx}
Based on your observations: "{turn1_response}"
Now complete your analysis:
1. How does the dialogue inform the emotion?
2. Resolve any conflicting cues
3. Make your final classification
Choose ONE from: {', '.join(EMOTION_NAMES)}.
{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}"""
