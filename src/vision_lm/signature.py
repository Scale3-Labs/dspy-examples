import dspy


class WebsiteDataExtractionSignature(dspy.Signature):
    """Website data extraction"""
    website_screenshot: dspy.Image = dspy.InputField(
        desc="A screenshot of the website"
    )
    hero_text: str = dspy.OutputField(
        desc="The hero text of the website"
    )
    website_description: str = dspy.OutputField(
        desc="A description of the website"
    )
    call_to_action: str = dspy.OutputField(
        desc="The call to action of the website"
    )
    color_palette: list[str] = dspy.OutputField(
        desc="The color palette of the website"
    )
    font_palette: list[str] = dspy.OutputField(
        desc="The font palette of the website"
    )
