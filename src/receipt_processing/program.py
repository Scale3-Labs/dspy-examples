import dspy

from signature import ReceiptDataExtractionSignature


class ReceiptDataExtraction(dspy.Module):
    def __init__(self):
        self.receipt_data_extraction = dspy.ChainOfThought(
            ReceiptDataExtractionSignature
        )

    def forward(self, receipt_screenshot: str):
        receipt_data = self.receipt_data_extraction(
            receipt_screenshot=receipt_screenshot
        )
        return receipt_data
