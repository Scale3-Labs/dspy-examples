
import base64
import dspy
from langtrace_python_sdk import langtrace
from program import WebsiteDataExtraction


langtrace.init()


def extract_website_data(website_screenshot_path: str):
    # Load the image
    with open(website_screenshot_path, "rb") as image_file:
        base64_data = base64.b64encode(
            image_file.read()
        ).decode('utf-8').replace('\n', '')
    image_data_uri = f"data:image/png;base64,{base64_data}"

    website_data_extraction = WebsiteDataExtraction()
    website_data = website_data_extraction(image_data_uri)
    return website_data


if __name__ == "__main__":

    dspy_lm = dspy.LM(model="openai/gpt-4o-mini")
    dspy.configure(lm=dspy_lm)

    result = extract_website_data(
        "src/vision_lm/data/langtrace-screenshot.png"
    )
    print(result)
