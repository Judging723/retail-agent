from fastapi import FastAPI


def test_conversation_routes_document_error_responses(
    application: FastAPI,
) -> None:
    paths = application.openapi()["paths"]

    create_responses = paths["/api/v1/conversations"]["post"]["responses"]

    detail_responses = paths["/api/v1/conversations/{conversation_no}"]["get"]["responses"]

    message_responses = paths["/api/v1/conversations/{conversation_no}/messages"]["post"][
        "responses"
    ]

    assert {"201", "422"} <= set(create_responses)
    assert {"200", "404"} <= set(detail_responses)
    assert {"201", "404", "409", "422"} <= set(message_responses)

    error_schema = detail_responses["404"]["content"]["application/json"]["schema"]

    assert error_schema == {"$ref": "#/components/schemas/ErrorResponse"}
