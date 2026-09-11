from app.services.report import build_report_html


def test_report_renders_experimental_native_class_without_probability_claim() -> None:
    result = {
        "exam_id": "demo-1",
        "tlc_profile_id": "client-device-tlc-pending",
        "source_images": 1,
        "plates_detected": 1,
        "bilateral_pairs_created": 0,
        "sources": [
            {
                "plates": [
                    {
                        "plate_id": "mouse-P01",
                        "image_url": "/static/generated/demo.png",
                        "tlc_profile_id": "client-device-tlc-pending",
                        "qc": {"status": "PASS", "flags": []},
                        "morphology_descriptor": "focal/irregular-region",
                        "native_binary_research": {
                            "available": True,
                            "research_binary_class": "TUMOR_LIKE",
                            "model_score": 0.61,
                            "validation_status": "NOT_VALIDATED_SINGLE_NEGATIVE",
                            "warning": "8 tumor / 1 no-tumor research fit",
                        },
                    }
                ]
            }
        ],
        "bilateral_analysis": [],
    }
    html = build_report_html(result)
    assert "Experimental native class" in html
    assert "TUMOR_LIKE" in html
    assert "Research model score" in html
    assert "NOT_VALIDATED_SINGLE_NEGATIVE" in html
    assert "not a cancer probability" in html
    assert "tumor-size estimate" in html
