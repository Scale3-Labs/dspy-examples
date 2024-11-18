import dspy

from signature import WebsiteDataExtractionSignature


class WebsiteDataExtraction(dspy.Module):
    def __init__(self):
        self.website_data_extraction = dspy.ChainOfThought(
            WebsiteDataExtractionSignature
        )

    def forward(self, website_screenshot: str):
        website_data = self.website_data_extraction(
            website_screenshot=website_screenshot
        )
        return website_data
