
class PromptTemplates:
    
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
    def _format_audio_block(audio_emotion: str, audio_confidence: float,
                             audio_analysis: str) -> str:
        return f"""AUDIO (external acoustic classifier):
{{
    "emotion": "{audio_emotion}",
    "source": "external_acoustic_classifier",
    "raw_description": "{audio_analysis}"
}}"""
    
    @staticmethod
    def direct(utterance: str, audio_analysis: str) -> str:
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        return f"""{ctx}
Analyze the video frames and classify the dominant emotion expressed.
Summarize your reasoning with few words.
Choose ONE from: {', '.join(EMOTION_NAMES)}.
Conclude with: '{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}'"""

    @staticmethod
    def chain_of_thought(utterance: str, audio_analysis: str) -> str:
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        return f"""{ctx}
Analyze the video frames step-by-step:
1. VISUAL: Facial expressions and body language across the video sequence
2. SPEECH: Dialogue content and vocal prosody
3. SYNTHESIS: Integrate visual cues with speech information

Summarize your reasoning with few words.
Choose ONE from: {', '.join(EMOTION_NAMES)}.
Conclude with: '{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}'"""

    @staticmethod
    def chain_of_thought_turn1(utterance: str, audio_analysis: str) -> str:
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        return f"""{ctx}
STEP-BY-STEP ANALYSIS (Part 1):
1. What facial expressions do you observe across the video?
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
Conclude with: '{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}'"""

    @staticmethod
    def visual_agent(utterance: str, audio_analysis: str) -> str:
        return f"""
You are a visual emotion expert. FOCUS PRIMARILY ON VISUAL CUES (facial expressions, body language, temporal evolution across frames).
Use audio/dialogue only as secondary context to help interpret what you see.
Analyze the video frames and classify the dominant emotion expressed.

DIALOGUE: "{utterance}"

Output format (REQUIRED):
Explanation: <Write 2-3 sentences explaining exactly what visual cues you see and why they lead to the emotion>
Emotion: <MUST BE EXACTLY ONE OF: {', '.join(EMOTION_NAMES)}.>
Conclude with: '{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}'
"""

    @staticmethod
    def speech_agent(utterance: str, audio_analysis: str, audio_emotion: str = "unknown", audio_confidence: float = 0.0) -> str:
        return f"""
You are a speech and pragmatic emotion expert. Analyze the alignment between the literal words and the acoustic delivery.

DIALOGUE: "{utterance}"
EXTERNAL ACOUSTIC CLASSIFIER: {audio_emotion} 
ACOUSTIC DETAILS: {audio_analysis}

Think about pragmatics: Does the tone match the words? Is it sarcasm? Is it genuine frustration or a lighthearted joke?

Output format (REQUIRED):
Explanation: <Write 2-3 sentences explaining the relationship between the words spoken and the tone of voice>
Emotion: <MUST BE EXACTLY ONE OF: {', '.join(EMOTION_NAMES)}>
Conclude with: '{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}'
"""

    @staticmethod
    def synthesizer(visual_result: str, speech_result: str,
                    utterance: str, audio_analysis: str,
                    audio_emotion: str = "unknown",
                    audio_confidence: float = 0.0) -> str:
        
        ctx = PromptTemplates._build_context(utterance, audio_analysis)
        audio_block = PromptTemplates._format_audio_block(
            audio_emotion, audio_confidence, audio_analysis
        )

        prompt = f"""{ctx}
You are the Lead Synthesis Expert. You have NO DIRECT ACCESS to video.
Your task is to determine the final true emotion of the character by reading the reports from your visual and speech agents.

=== VISUAL AGENT REPORT ===
{visual_result}

=== SPEECH AGENT REPORT ===
{speech_result}

INSTRUCTIONS:
1. Read both reports carefully. Ignore any confidence scores. Focus only on the explanations.
3. If they agree, use both to support your decision.
4. PRIORITY POLICY: 
 4.1. Prioritize speech if visual is not different sentiment polarity. 
 4.2. Explicitly compare the two explanations and identify which one has stronger evidence emotions to reavaluate the emotion. Compare Specificity, Coherence and Contextual plausibility.
5. You MUST choose exactly ONE emotion from: anger, disgust, fear, joy, neutral, sadness, surprise.

Output format:
Explanation: <Briefly explain how you resolved the inputs from Visual and Speech into your final decision>
Emotion: <MUST BE EXACTLY ONE OF: {', '.join(EMOTION_NAMES)}>

Conclude with: '{FINAL_TAG_START}Conclusion: the dominant emotion is EMOTION.{FINAL_TAG_END}' """
        return prompt
