"""JSON schema for the lesson plan. This doubles as the tool input schema handed
to the model, so the shape is enforced by the API rather than by parsing prose."""

from app.models import MODULE_TITLES

LESSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "gap_analysis": {
            "type": "string",
            "description": (
                "2-4 sentences mapping the role's requirements against the "
                "candidate's resume: what matches, what is missing, what is adjacent."
            ),
        },
        "modules": {
            "type": "array",
            "description": f"Exactly {len(MODULE_TITLES)} modules, in the given order.",
            "minItems": len(MODULE_TITLES),
            "maxItems": len(MODULE_TITLES),
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string", "enum": MODULE_TITLES},
                    "content": {
                        "type": "string",
                        "description": (
                            "The teaching content for this module, grounded strictly "
                            "in the supplied evidence. Cite concrete facts, not "
                            "generic advice."
                        ),
                    },
                    "quiz": {
                        "type": "array",
                        "description": "2-3 questions answerable from this module's content.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "expected_points": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Key points a correct answer covers.",
                                },
                            },
                            "required": ["question", "expected_points"],
                        },
                    },
                },
                "required": ["number", "title", "content", "quiz"],
            },
        },
    },
    "required": ["gap_analysis", "modules"],
}
