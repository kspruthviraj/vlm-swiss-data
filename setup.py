from setuptools import setup, find_packages

setup(
    name="vlm-swiss-data",
    version="0.1.0",
    description="Multimodal Swiss document dataset for VLM fine-tuning",
    author="Sreenath",
    python_requires=">=3.10",
    packages=find_packages(),
    py_modules=["collect"],
    install_requires=[
        "httpx>=0.25.0",
        "datasets>=2.14.0",
        "Pillow>=10.0.0",
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.10.0",
        "huggingface-hub>=0.17.0",
    ],
    entry_points={
        "console_scripts": [
            "swiss-collect=collect:main",
        ],
    },
)
