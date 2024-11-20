import dspy


class ReceiptDataExtractionSignature(dspy.Signature):
    """Receipt data extraction"""
    receipt_screenshot: dspy.Image = dspy.InputField(
        desc="A screenshot of the receipt"
    )
    vendor_name: str = dspy.OutputField(
        desc="The name of the vendor"
    )
    purchase_date: str = dspy.OutputField(
        desc="The date of the purchase"
    )
    purchase_time: str = dspy.OutputField(
        desc="The time of the purchase"
    )
    items: list[dict] = dspy.OutputField(
        desc="The items purchased"
    )
    total_amount: float = dspy.OutputField(
        desc="The total amount of the purchase"
    )