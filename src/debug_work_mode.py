from src.extractor import JobAttributeExtractor

text = """
InnovateTech Solutions Pvt Ltd is hiring an AI Engineer
in Hyderabad. Candidates must have a B.Tech degree and
2-4 years of experience in machine learning.
Required skills include Python, TensorFlow, SQL, AWS,
and Docker. This position offers a hybrid work mode
with an annual salary of 10-15 LPA.
"""

extractor = JobAttributeExtractor()

for item in extractor.predict(text):
    word = text[item["start"]:item["end"]]

    if any(term in word.lower() for term in ["hybrid", "work", "mode"]):
        print(
            f"Text: {word!r:15} "
            f"Label: {item['label']:18} "
            f"Confidence: {item['confidence']:.4f}"
        )
