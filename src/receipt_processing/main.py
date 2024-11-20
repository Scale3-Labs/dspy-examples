import dspy
import base64
from langtrace_python_sdk import langtrace
from custom_lm import CustomLM
from program import ReceiptDataExtraction
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

langtrace.init()


def extract_receipt_data(receipt_screenshot_path: str):
    # Load the image
    with open(receipt_screenshot_path, "rb") as image_file:
        base64_data = base64.b64encode(
            image_file.read()
        ).decode('utf-8').replace('\n', '')
    image_data_uri = f"data:image/png;base64,{base64_data}"

    receipt_data_extraction = ReceiptDataExtraction()
    receipt_data = receipt_data_extraction(image_data_uri)
    return receipt_data


if __name__ == "__main__":
    # Get the tracer
    tracer = trace.get_tracer(__name__)

    dspy_lm = CustomLM(model="openai/gpt-4o-mini", cache=False, logprobs=True)
    dspy.configure(lm=dspy_lm)

    with tracer.start_as_current_span("receipt_processing") as span:
        result = extract_receipt_data(
            "src/receipt_processing/data/example-1.png"
        )
        accuracy = dspy_lm.get_accuracy()
        # Add accuracy as an attribute to the span
        span.set_attribute("model.accuracy", str(accuracy))
        span.set_status(Status(StatusCode.OK))
 
    print("\n")
    print("Accuracy: ", accuracy)
    print("\n")
    print("Result: ", result)
