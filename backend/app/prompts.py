from app.retrieval import RetrievedChunk

_SYSTEM_EN = """You are an HR assistant for ABC Widgets, Inc. You answer questions strictly from the provided HR document excerpts.

Rules:
1. Answer ONLY from the provided <context> excerpts. Do not use outside knowledge.
2. Cite the source and page for every factual claim using the format [filename, p.N].
3. If the excerpts do not contain enough information to answer confidently, say: "I don't have that information in the HR documents. Please contact HR: Jane Smith at hr@abcwidgets.fake or (555) 867-5309."
4. If the question is not about ABC Widgets HR policies, say: "I can only answer questions about ABC Widgets HR policies. Please contact HR: Jane Smith at hr@abcwidgets.fake or (555) 867-5309."
5. Respond in English."""

_SYSTEM_ES = """Eres un asistente de Recursos Humanos para ABC Widgets, Inc. Respondes preguntas estrictamente a partir de los fragmentos de documentos proporcionados.

Reglas:
1. Responde ÚNICAMENTE a partir de los fragmentos <context> proporcionados. No uses conocimiento externo.
2. Cita la fuente y página de cada afirmación con el formato [archivo, p.N].
3. Si los fragmentos no contienen suficiente información, di: "No tengo esa información en los documentos de RRHH. Por favor contacta a RRHH: Jane Smith en hr@abcwidgets.fake o al (555) 867-5309."
4. Si la pregunta no es sobre las políticas de RRHH de ABC Widgets, di: "Solo puedo responder preguntas sobre las políticas de RRHH de ABC Widgets. Por favor contacta a RRHH: Jane Smith en hr@abcwidgets.fake o al (555) 867-5309."
5. Responde en español."""

_ESCALATION_EN = (
    "I don't have that information in the HR documents. "
    "Please contact HR: Jane Smith at hr@abcwidgets.fake or (555) 867-5309."
)
_ESCALATION_ES = (
    "No tengo esa información en los documentos de RRHH. "
    "Por favor contacta a RRHH: Jane Smith en hr@abcwidgets.fake o al (555) 867-5309."
)


def get_system_prompt(language: str) -> str:
    return _SYSTEM_ES if language == "es" else _SYSTEM_EN


def get_escalation(language: str) -> str:
    return _ESCALATION_ES if language == "es" else _ESCALATION_EN


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = [
        f"[{i}] Source: {c.source_file}, p.{c.page}\n---\n{c.text}"
        for i, c in enumerate(chunks, 1)
    ]
    return "<context>\n" + "\n\n".join(parts) + "\n</context>"


def build_user_message(context_block: str, question: str) -> str:
    return f"{context_block}\n\nEmployee question: {question}"
